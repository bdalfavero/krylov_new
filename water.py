import json
import argparse
import h5py
import numpy as np
from scipy.sparse.linalg import norm
import pandas as pd
import openfermion as of
from openfermionpyscf import run_pyscf
import quimb
from fermion_helpers import fock_state
from convert import cirq_pauli_sum_to_qiskit_pauli_op
from tensor_network_common import pauli_sum_to_mpo
from krylov import (
    generate_u_subspace,
    generate_h_subspace,
    fill_subspace_matrices_full,
    fill_subspace_matrices_toeplitz,
    toeplitz_elements_from_vectors,
    energy_vs_d
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", type=str, help="HDF5 file for program output.")
    args = parser.parse_args()

    n_elec = 10
    with open("data/water_params.json", "r") as f:
        input_dict = json.load(f)
    d = input_dict["d"]
    eps = input_dict["eps"]

    hamiltonian_file = "data/monomer_eqb"
    hamiltonian = of.jordan_wigner(
            of.get_fermion_operator(
        of.chem.MolecularData(filename=hamiltonian_file).get_molecular_hamiltonian()
        )
    )
    ham_cirq = of.transforms.qubit_operator_to_pauli_sum(hamiltonian)
    qs = ham_cirq.qubits
    nq = len(qs)
    # ham_qiskit = cirq_pauli_sum_to_qiskit_pauli_op(ham_cirq)
    # ham_mpo = quimb.load_from_disk("data/water_mpo.data")
    ham_sparse = of.linalg.get_sparse_operator(hamiltonian)
    tau = np.pi / norm(ham_sparse)
    print(f"tau = {tau}")

    ref_state = fock_state([True] * n_elec + [False] * (nq - n_elec))
    states = generate_u_subspace(ref_state, ham_sparse, tau, d)
    h, s = fill_subspace_matrices_full(ham_sparse, states)
    ds, energies, num_kept = energy_vs_d(h, s, eps)

    for dd, energ in zip(ds, energies):
        print(f"{dd} {energies}")

    f = h5py.File(args.output_file, "w")
    f.create_dataset("tau", data=tau)
    f.create_dataset("ds", data=ds)
    f.create_dataset("energies", data=energies)
    f.create_dataset("num_kept", data=num_kept)
    f.close()

if __name__ == "__main__":
    main()