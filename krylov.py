from typing import List, Tuple
import numpy as np
import scipy.linalg as la

def generate_u_subspace(
    psi: np.ndarray, hamiltonian: np.ndarray, t: float, d: int
) -> List[np.ndarray]:
    """Get the subspace spanned by {psi, U phi, U^2 psi, ... U^(d-1) phi},
    where U = exp(-i H t). U is computed exactly (no Trotter!)."""

    u = expm(-1j * t * hamiltonian)
    psi_evolved = psi.copy()
    states: List[np.ndarray] = []
    for i in range(d):
        states.append(psi_evolved.copy())
        if i != d - 1:
            psi_evolved = u @ psi_evolved
            psi_evolved = psi_evolved / la.norm(psi_evolved)
    return states


def fill_subspace_matrices_toeplitz(
    mat_elems: List[complex], overlaps: List[complex]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill subspace matrices from the computed matrix elements and overlaps."""

    assert len(mat_elems) == len(overlaps)
    d = len(mat_elems)
    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    for i in range(d): # Loop over rows.
        for j in range(d):
            if i >= j:
                h[i, j] = mat_elems[i - j].conjugate()
                s[i, j] = overlaps[i - j].conjugate()
            else: # i < j
                h[i, j] = mat_elems[j - i]
                s[i, j] = overlaps[j - i]
    return h, s


def fill_subspace_matrices_full(
    hamiltonian: np.ndarray, states: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the subspace matrices from a list of states."""

    d = len(states)
    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    for i in range(d): # Loop over rows.
        for j in range(d):
            h[i, j] = np.vdot(states[i], hamiltonian @ states[j])
            s[i, j] = np.vdot(states[i], states[j])
    return (h, s)


def threshold_eigenvalues(h: np.ndarray, s: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Remove all eigenvalues below a positive threshold eps.
    See Epperly et al. sec. 1.2."""

    # Build a matrix whose columns correspond to the positive eigenvectors of s.
    evals, evecs = la.eigh(s)
    positive_evals = []
    positive_evecs = []
    for i, ev in enumerate(evals):
        assert abs(ev.imag) < 1e-7
        if ev.real > eps:
            positive_evals.append(ev.real)
            positive_evecs.append(evecs[:, i])
    pos_evec_mat = np.vstack(positive_evecs).T
    # Project h and s into this subspace.
    new_s =  pos_evec_mat.conj().T @ s @ pos_evec_mat
    new_h = pos_evec_mat.conj().T @ h @ pos_evec_mat
    return new_h, new_s


def energy_vs_d(h, s, eps: float, step:int = 1) -> Tuple[List[int], List[float]]:
    """Get energy vs. subspace dimension.
    If H and S are of dimension D, we can get the energy estimate
    for d < D by taking the upper left d x d blocks of H and S."""

    assert h.shape == s.shape
    assert h.shape[0] == h.shape[1]
    ds: List[int] = []
    energies: List[float] = []
    for d in range(1, h.shape[0], step):
        h_d, s_d = h[:d, :d], s[:d, :d]
        ds.append(d)
        ht, st = threshold_eigenvalues(h_d, s_d, eps)
        lam, v = la.eig(ht, st)
        energies.append(np.min(lam).real)
    return ds, energies