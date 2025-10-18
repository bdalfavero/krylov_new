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
    l = 2
    t = 1.0
    u = 4.0
    n_elec = 2
    alpha = 10.0 # For number enforcement

    hamiltonian = of.hamiltonians.fermi_hubbard(l, l, t, u, spinless=True)
    ham_jw = of.transforms.jordan_wigner(hamiltonian)
    ham_augmented = add_number_term(ham_jw, n_elec, alpha)
    ham_sparse = of.linalg.get_sparse_operator(ham_jw)

    eigvals, eigvecs = eigsh(ham_sparse, which='SA')
    i_min = np.argmin(eigvals.real)
    ground_energy = eigvals[i_min]
    ground_state = eigvecs[:, i_min]
    print(f"Ground energy = {ground_energy}")
    np.save("data/ground_state.npy", ground_state)

if __name__ == "__main__":
    main()
