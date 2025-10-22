import json
import openfermion as of
import numpy as np
from scipy.sparse.linalg import eigsh
import openfermion as of
from krylov import (
    fill_subspace_matrices_full,
    threshold_eigenvalues, 
    energy_vs_d
)
from fermion_helpers import add_number_term

def main():
    with open("data/hubbard_params.json", "r") as f:
        input_dict = json.load(f)
    l = input_dict["l"]
    t = input_dict["t"]
    u = input_dict["u"]
    n_elec = input_dict["n_elec"]
    alpha = input_dict["alpha"]

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    ham_jw = of.transforms.jordan_wigner(hamiltonian)
    ham_augmented = add_number_term(ham_jw, n_elec, alpha)
    ham_sparse = of.linalg.get_sparse_operator(ham_augmented)

    eigvals, eigvecs = eigsh(ham_sparse, which='SA')
    i_min = np.argmin(eigvals.real)
    ground_energy = eigvals[i_min]
    ground_state = eigvecs[:, i_min]
    print(f"Ground energy = {ground_energy}")
    np.save("data/ground_state.npy", ground_state)
    np.save("data/ground_energy.npy", ground_energy)

if __name__ == "__main__":
    main()
