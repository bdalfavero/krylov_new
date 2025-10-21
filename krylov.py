from typing import List, Tuple
import numpy as np
import scipy.linalg as la
import qiskit
from qiskit.qasm2 import dumps
from qiskit_aer import AerSimulator
import quimb
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
from scipy.sparse.linalg import expm

def generate_h_subspace(
    psi: np.ndarray, hamiltonian: np.ndarray, d: int
) -> List[np.ndarray]:
    """Get the subspace spanned by {psi, H phi, H^2 psi, ... H^(d-1) phi},
    where each state is normalized."""

    psi_evolved = psi.copy()
    states: List[np.ndarray] = []
    for i in range(d):
        states.append(psi_evolved.copy())
        if i != d - 1:
            psi_evolved = hamiltonian @ psi_evolved
            psi_evolved = psi_evolved / la.norm(psi_evolved)
    return states


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


def _evolve_state_qiskit(
        reference_state: np.ndarray, evolution_circuit: qiskit.QuantumCircuit, nq: int 
) -> np.ndarray:
    """Get the state vector corresponding to (U_evolution)^d U_prep |0>."""

    sim = AerSimulator(method="statevector")
    total_circuit = qiskit.QuantumCircuit(nq)
    total_circuit.initialize(reference_state)
    total_circuit = total_circuit.compose(evolution_circuit)
    transpiled_circuit = qiskit.transpile(total_circuit, sim)
    transpiled_circuit.save_state()
    result = sim.run(transpiled_circuit).result()
    sv = result.get_statevector()
    return sv.data


def generate_circuit_subspace(
    psi: np.ndarray, ev_circuit: qiskit.QuantumCircuit, t: float, d: int, nq
) -> List[np.ndarray]:
    """Get the subspace spanned by {psi, U phi, U^2 psi, ... U^(d-1) phi},
    where U = exp(-i H t). U is computed exactly (no Trotter!)."""

    psi_evolved = psi.copy()
    states: List[np.ndarray] = []
    for i in range(d):
        states.append(psi_evolved.copy())
        if i != d - 1:
            psi_evolved = _evolve_state_qiskit(psi_evolved, ev_circuit, nq)
    return states


def toeplitz_elements_from_vectors(
    hamiltoian: np.ndarray, states: List[np.ndarray]
) -> Tuple[List[complex], List[complex]]:
    """Generates <psi_0| H |psi_i> and <psi_0|psi_i> from the list of states
    {psi_0, psi_1, ..., psi_L} and the Hamiltonian H."""

    overlaps: List[complex] = []
    mat_elems: List[complex] = []
    for psi in states:
        overlaps.append(np.vdot(states[0], psi))
        mat_elems.append(np.vdot(states[0], hamiltoian @ psi))
    return mat_elems, overlaps


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


def tebd_states_to_scratch(
    ev_circuit: qiskit.QuantumCircuit,
    ref_state: MatrixProductState, max_bond: int, d: int,
    scratch_dir: str, backend_callback
) -> List[str]:
    """Do successive steps of TEBD with the same circuit, storing the intermediate MPS's
    in a scratch directory."""

    qasm_str = dumps(ev_circuit)
    d_path_dict: List[str] = []
    evolved_mps = ref_state.copy()
    for i in range(d):
        print(f"d = {i}")
        fname = f"{scratch_dir}/state_{i}.dump"
        quimb.save_to_disk(evolved_mps, fname)
        d_path_dict.append(fname)
        if i != d - 1:
            if backend_callback is not None:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_bond, progbar=False,
                    to_backend=backend_callback
                )
            else:
                circuit_mps = qtn.circuit.CircuitMPS.from_openqasm2_str(
                    qasm_str, psi0=evolved_mps, max_bond=max_bond, progbar=False
                )
            evolved_mps = circuit_mps.psi
    return d_path_dict


def toeplitz_elements_from_files(
    fnames: List[str], ham_mpo: MatrixProductOperator
) -> Tuple[List[complex], List[complex]]:
    """Get matrix elements and overlaps using the stored states."""

    overlaps: List[complex] = []
    mat_elems: List[complex] = []
    state_0 = quimb.load_from_disk(fnames[0])
    for i in range(len(fnames)):
        state_i = quimb.load_from_disk(fnames[i])
        overlaps.append(state_0.H @ state_i)
        mat_elems.append(state_0.H @ ham_mpo.apply(state_i))
    return (mat_elems, overlaps)


def fill_subspace_matrices_from_fname_dict(
    fname_dict: List[str], ham_mpo: MatrixProductOperator, d: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill subspace matrices from a dictionary mapping integers (the power of the unitary)
    to the filename where the MPS is stored."""

    h = np.zeros((d, d), dtype=complex)
    s = np.zeros((d, d), dtype=complex)
    # for i in range(d):
    #     state_i = quimb.load_from_disk(fname_dict[i])
    #     for j in range(i+1, d):
    #         state_j = quimb.load_from_disk(fname_dict[j])
    #         h[i, j] = state_i.H @ ham_mpo.apply(state_j)
    #         s[i, j] = state_i.H @ state_j
    # h += h.conj().T
    # s += s.conj().T
    # for i in range(d):
    #     state_i = quimb.load_from_disk(fname_dict[i])
    #     h[i, i] = state_i.H @ ham_mpo.apply(state_i)
    #     s[i, i] = state_i.H @ state_i
    for i in range(d):
        state_i = quimb.load_from_disk(fname_dict[i])
        for j in range(d):
            state_j = quimb.load_from_disk(fname_dict[j])
            if i >= j:
                h[i, j] = (state_i.H @ ham_mpo.apply(state_j)).conjugate()
                s[i, j] = (state_i.H @ state_j).conjugate()
            else:
                h[i, j] = state_i.H @ ham_mpo.apply(state_j)
                s[i, j] = state_i.H @ state_j
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


def energy_vs_d(h, s, eps: float, step:int = 1) -> Tuple[List[int], List[float], List[int]]:
    """Get energy vs. subspace dimension.
    If H and S are of dimension D, we can get the energy estimate
    for d < D by taking the upper left d x d blocks of H and S."""

    assert h.shape == s.shape
    assert h.shape[0] == h.shape[1]
    ds: List[int] = []
    energies: List[float] = []
    num_kept: List[int] = []
    for d in range(1, h.shape[0], step):
        h_d, s_d = h[:d, :d], s[:d, :d]
        ds.append(d)
        ht, st = threshold_eigenvalues(h_d, s_d, eps)
        num_kept.append(ht.shape[0])
        lam, v = la.eig(ht, st)
        energies.append(np.min(lam).real)
    return ds, energies, num_kept
