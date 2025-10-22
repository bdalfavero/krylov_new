import argparse
from typing import List
import json
import numpy as np
import scipy.linalg as la
import pandas as pd
from scipy.sparse.linalg import norm
import openfermion as of
from krylov import (
    generate_u_subspace,
    generate_h_subspace,
    toeplitz_elements_from_vectors,
    fill_subspace_matrices_full,
    fill_subspace_matrices_toeplitz,
    energy_vs_d
)
from fermion_helpers import add_number_term, fock_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method to use.")
    parser.add_argument("output_file", type=str, help="CSV file with energy vs. d.")
    args = parser.parse_args()

    with open("data/hubbard_params.json", "r") as f:
        input_dict = json.load(f)
    l = input_dict["l"]
    t = input_dict["t"]
    u = input_dict["u"]
    n_elec = input_dict["n_elec"]
    alpha = input_dict["alpha"]
    d = input_dict["d"]
    eps = input_dict["eps"]

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    ham_jw = of.transforms.jordan_wigner(hamiltonian)
    ham_augmented = add_number_term(ham_jw, n_elec, alpha)
    ham_sparse = of.linalg.get_sparse_operator(ham_jw)

    tau = np.pi / norm(ham_sparse)

    # We will use a Neel state as our reference.
    nq = of.utils.count_qubits(ham_augmented)
    bools = [True, False, True, False]
    assert len(bools) == nq
    assert np.sum(bools) == n_elec
    psi = fock_state(bools)

    valid_methods = ["H", "U", "Toeplitz"]
    if args.method == "H":
        states = generate_h_subspace(psi, ham_sparse, d)
        h, s = fill_subspace_matrices_full(ham_sparse, states)
    elif args.method == "U":
        states = generate_u_subspace(psi, ham_sparse, tau, d)
        h, s = fill_subspace_matrices_full(ham_sparse, states)
    elif args.method == "Toeplitz":
        states = generate_u_subspace(psi, ham_sparse, tau, d)
        mat_elems, overlaps = toeplitz_elements_from_vectors(ham_sparse, states)
        h, s = fill_subspace_matrices_toeplitz(mat_elems, overlaps)
    else:
        raise ValueError(f"Unrecognized method {args.method}.")

    ds, energies = energy_vs_d(h, s, eps)
    for d, ener in zip(ds, energies):
        print(f"{d} {ener}")

    df = pd.DataFrame({"d": ds, "energy": energies})
    df.set_index("d", inplace=True)
    df.index.name = "d"
    df.to_csv(args.output_file)

if __name__ == "__main__":
    main()
