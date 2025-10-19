from typing import List
import json
import numpy as np
import scipy.linalg as la
import pandas as pd
import openfermion as of
from krylov import (
    generate_h_subspace,
    fill_subspace_matrices_full,
    energy_vs_d
)
from fermion_helpers import add_number_term, fock_state

def main():
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

    # We will use a Neel state as our reference.
    nq = of.utils.count_qubits(ham_augmented)
    bools = [True, False, True, False]
    assert len(bools) == nq
    assert np.sum(bools) == n_elec
    psi = fock_state(bools)
    # psi = np.load("data/ground_state.npy")
    states = generate_h_subspace(psi, ham_sparse, d)
    h, s = fill_subspace_matrices_full(ham_sparse, states)
    ds, energies = energy_vs_d(h, s, eps)
    for d, ener in zip(ds, energies):
        print(f"{d} {ener}")
    
    df = pd.DataFrame({"d": ds, "energy": energies})
    df.set_index("d", inplace=True)
    df.index.name = "d"
    df.to_csv("data/hubbard_h.csv")

if __name__ == "__main__":
    main()
