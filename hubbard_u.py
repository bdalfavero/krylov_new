from typing import List
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import expm, norm
import openfermion as of
from krylov import (
    fill_subspace_matrices_full,
    threshold_eigenvalues, 
    energy_vs_d
)
from fermion_helpers import add_number_term, fock_state

def generate_u_subspace(
    psi: np.ndarray, hamiltonian: np.ndarray, t: float, d: int
) -> List[np.ndarray]:
    """Get the subspace spanned by {psi, U phi, U^2 psi, ... U^(d-1) phi},
    where U = exp(-i H t)."""

    u = expm(-1j * t * hamiltonian)
    psi_evolved = psi.copy()
    states: List[np.ndarray] = []
    for i in range(d):
        states.append(psi_evolved.copy())
        if i != d - 1:
            psi_evolved = u @ psi_evolved
            psi_evolved = psi_evolved / la.norm(psi_evolved)
    return states


def main():
    l = 2
    t = 1.0
    u = 4.0
    n_elec = 2
    alpha = 1.0 # For number enforcement
    eps = 1e-8
    d = 50

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
    # psi = np.load("data/ground_state.npy")
    states = generate_u_subspace(psi, ham_sparse, tau, d)
    h, s = fill_subspace_matrices_full(ham_sparse, states)
    ds, energies = energy_vs_d(h, s, eps)
    for d, ener in zip(ds, energies):
        print(f"{d} {ener}")

if __name__ == "__main__":
    main()
