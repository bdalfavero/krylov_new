import argparse
from typing import List
import json
import numpy as np
import scipy.linalg as la
import pandas as pd
from scipy.sparse.linalg import norm
from quimb.tensor.tensor_1d import MatrixProductState
import cirq
import openfermion as of
import qiskit
from fermion_helpers import add_number_term, fock_state
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from tensor_network_common import pauli_sum_to_mpo
from krylov import (
    tebd_states_to_scratch,
    fill_subspace_matrices_from_fname_dict,
    toeplitz_elements_from_files,
    fill_subspace_matrices_toeplitz,
    energy_vs_d
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method to fill the matrices (full or Toeplitz)")
    parser.add_argument("output_file", type=str, help="CSV file with energy vs. d.")
    args = parser.parse_args()

    max_mpo_bond = 100
    max_tebd_bond = 15

    with open("data/hubbard_params.json", "r") as f:
        input_dict = json.load(f)
    l = input_dict["l"]
    t = input_dict["t"]
    u = input_dict["u"]
    n_elec = input_dict["n_elec"]
    alpha = input_dict["alpha"]
    d = input_dict["d"]
    eps = input_dict["eps"]
    steps = input_dict["steps"]

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    ham_jw = of.transforms.jordan_wigner(hamiltonian)
    nq = of.utils.count_qubits(ham_jw)
    qs = cirq.LineQubit.range(nq)
    ham_augmented = add_number_term(ham_jw, n_elec, alpha)
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(ham_jw)
    ham_sparse = of.linalg.get_sparse_operator(ham_jw)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)

    tau = np.pi / norm(ham_sparse)

    # We will use a Neel state as our reference.
    nq = of.utils.count_qubits(ham_augmented)
    bools = [True, False, True, False]
    assert len(bools) == nq
    assert np.sum(bools) == n_elec
    psi = fock_state(bools)
    ref_state_mps = MatrixProductState.from_dense(psi)

    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))
    ev_circuit_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
    qiskit.qasm2.dump(ev_circuit_transpiled, "data/lih_circuit.qasm")

    scratch_dir = "data/hubbard_scratch"
    fnames = tebd_states_to_scratch(
        ev_circuit_transpiled, ref_state_mps, max_tebd_bond, d, scratch_dir, None
    )
    if args.method == "full":
        h, s = fill_subspace_matrices_from_fname_dict(fnames, ham_mpo, d)
    elif args.method == "Toeplitz":
        mat_elems, overlaps = toeplitz_elements_from_files(fnames, ham_mpo)
        h, s = fill_subspace_matrices_toeplitz(mat_elems, overlaps)
    else:
        raise ValueError(f"Unrecognized method {args.method}. Must be full or Toeplitz.")
    np.save("data/hubbard_h", h)
    np.save("data/hubbard_s", s)

    ds, energies = energy_vs_d(h, s, eps)
    for dd, ener in zip(ds, energies):
        print(f"{dd} {ener}")
    
    df = pd.DataFrame({"d": ds, "energy": energies})
    df.set_index("d", inplace=True)
    df.index.name = "d"
    df.to_csv(args.output_file)

if __name__ == "__main__":
    main()