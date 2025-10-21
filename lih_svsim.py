import argparse
import h5py
import numpy as np
from scipy.sparse.linalg import norm
import pandas as pd
import openfermion as of
import qiskit
from openfermionpyscf import run_pyscf
from fermion_helpers import add_number_term, fock_state
from krylov import (
    generate_circuit_subspace,
    fill_subspace_matrices_full,
    fill_subspace_matrices_toeplitz,
    toeplitz_elements_from_vectors,
    energy_vs_d
)
from convert import cirq_pauli_sum_to_qiskit_pauli_op
# for debug
from tensor_network_common import pauli_sum_to_mpo
import cirq
from quimb import save_to_disk
from quimb.tensor.tensor_1d import MatrixProductState
from krylov import fill_subspace_matrices_from_fname_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Krylov method to use.")
    parser.add_argument("output_file", type=str, help= "CSV file for output.")
    args = parser.parse_args()

    molec = "LiH"
    basis = "sto-3g"
    n_elec = 4
    alpha = 1.0
    d = 60
    eps = 1e-8
    steps = 20

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
    ham_augmented = add_number_term(hamiltonian_qubop, n_elec, alpha)
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian_qubop)
    ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    ham_sparse = of.linalg.get_sparse_operator(hamiltonian_qubop)
    tau = np.pi / norm(ham_sparse)

    dt = tau / float(steps)
    ev_gate = qiskit.circuit.library.PauliEvolutionGate(ham_qiskit, time=dt)
    ev_ckt_qiskit = qiskit.QuantumCircuit(nq)
    for _ in range(steps):
        ev_ckt_qiskit.append(ev_gate, range(nq))
    ev_circuit_transpiled = qiskit.transpile(ev_ckt_qiskit, basis_gates=["u3", "cx"])
    qiskit.qasm2.dump(ev_circuit_transpiled, "data/lih_circuit.qasm")

    ref_state = fock_state([True] * n_elec + [False] * (nq - n_elec))
    states = generate_circuit_subspace(ref_state, ev_circuit_transpiled, tau, d, nq)
    if args.method == "U":
        # TODO Try converting the states to MPSs and then fill from scratch.
        # h, s = fill_subspace_matrices_full(ham_sparse, states)
        qs = cirq.LineQubit.range(nq)
        max_mpo_bond = 100
        ham_mpo = pauli_sum_to_mpo(ham_cirq, qs, max_mpo_bond)
        scratch_dir = "data/lih_scratch_new"
        fnames = []
        for i, state in enumerate(states):
            mps = MatrixProductState.from_dense(state)
            fname = f"{scratch_dir}/state_{i}.dump"
            fnames.append(fname)
            save_to_disk(mps, fname)
        h, s = fill_subspace_matrices_from_fname_dict(fnames, ham_mpo, d)
    elif args.method == "Toeplitz":
        mat_elems, overlaps = toeplitz_elements_from_vectors(ham_sparse, states)
        h, s = fill_subspace_matrices_toeplitz(mat_elems, overlaps)
    ds, energies, num_kept = energy_vs_d(h, s, eps)
    for dd, ener, nkept in zip(ds, energies, num_kept):
        print(f"{dd} {ener} {nkept}")
    
    # df = pd.DataFrame({"d": ds, "energy": energies})
    # df.set_index("d", inplace=True)
    # df.index.name = "d"
    # df.to_csv(args.output_file)

    f = h5py.File(args.output_file, "w")
    f.create_dataset("eps", data=eps)
    f.create_dataset("tau", data=tau)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("ds", data=np.array(ds))
    f.create_dataset("energies", data=np.array(energies))
    f.create_dataset("num_kept", data=np.array(num_kept))
    f.close()

if __name__ == "__main__":
    main()