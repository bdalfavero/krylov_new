import argparse
import numpy as np
from scipy.sparse.linalg import norm
import pandas as pd
import cirq
import openfermion as of
import qiskit
from openfermionpyscf import run_pyscf
from quimb.tensor.tensor_1d import MatrixProductState
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from fermion_helpers import add_number_term, fock_state
from tensor_network_common import pauli_sum_to_mpo
from krylov import (
    tebd_states_to_scratch,
    fill_subspace_matrices_from_fname_dict,
    energy_vs_d
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str, help= "CSV file for output.")
    args = parser.parse_args()

    molec = "LiH"
    basis = "sto-3g"
    n_elec = 4
    alpha = 1.0
    d = 20
    eps = 1e-8
    max_mpo_bond = 100
    max_tebd_bond = 15
    steps = 10

    geometry = of.chem.geometry_from_pubchem(molec)
    multiplicity = 1
    molecule = of.chem.MolecularData(
        geometry, basis, multiplicity
    )
    molecule = run_pyscf(molecule, run_scf=1, run_fci=1)
    print(f"HF energy:", molecule.hf_energy)
    print(f"FCI energy:", molecule.fci_energy)
    hamiltonian = molecule.get_molecular_hamiltonian()
    hamiltonian_qubop = of.transforms.jordan_wigner(hamiltonian)
    nq = of.utils.count_qubits(hamiltonian_qubop)
    qs = cirq.LineQubit.range(nq)
    ham_augmented = add_number_term(hamiltonian_qubop, n_elec, alpha)
    ham_sparse = of.linalg.get_sparse_operator(hamiltonian_qubop)
    tau = np.pi / norm(ham_sparse)
    print(f"tau = {tau:4.5e}")
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop, qs)
    ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)

    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))
    ev_circuit_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
    qiskit.qasm2.dump(ev_circuit_transpiled, "data/lih_circuit.qasm")

    ref_state = fock_state([True] * n_elec + [False] * (nq - n_elec))
    ref_mps = MatrixProductState.from_dense(ref_state)
    scratch_dir = "data/lih_scratch"
    fnames = tebd_states_to_scratch(
        ev_circuit_transpiled, ref_mps, max_tebd_bond, d, scratch_dir, None
    )
    # TODO Checking abs(s[:, 0]), the state seems not to be evolving.
    h, s = fill_subspace_matrices_from_fname_dict(fnames, ham_mpo, d)
    np.save("data/lih_h", h)
    np.save("data/lih_s", s)

    ds, energies = energy_vs_d(h, s, eps)
    for dd, ener in zip(ds, energies):
        print(f"{dd} {ener}")
    
    df = pd.DataFrame({"d": ds, "energy": energies})
    df.set_index("d", inplace=True)
    df.index.name = "d"
    df.to_csv(args.output_file)

if __name__ == "__main__":
    main()