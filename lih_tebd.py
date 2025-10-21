import argparse
import h5py
import numpy as np
from scipy.sparse.linalg import norm
import pandas as pd
import cirq
import openfermion as of
import qiskit
from qiskit.qasm2 import dumps
from openfermionpyscf import run_pyscf
from quimb.tensor.tensor_1d import MatrixProductState
from quimb.tensor.circuit import CircuitMPS
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
    d = 60
    eps = 1e-8
    max_mpo_bond = 100
    max_tebd_bond = 50
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
    # tau = 0.1
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

    reference_circuit = qiskit.QuantumCircuit(nq)
    for i in range(nq):
        if i < n_elec:
            reference_circuit.x(i)
    ref_circuit_qasm = dumps(reference_circuit)
    quimb_circuit = CircuitMPS.from_openqasm2_str(ref_circuit_qasm)
    reference_mps = quimb_circuit.psi

    # ref_state = fock_state([True] * n_elec + [False] * (nq - n_elec))
    # ref_mps = MatrixProductState.from_dense(ref_state)

    scratch_dir = "data/lih_scratch"
    fnames = tebd_states_to_scratch(
        ev_circuit_transpiled, reference_mps, max_tebd_bond, d, scratch_dir, None
    )
    # TODO Checking abs(s[:, 0]), the state seems not to be evolving.
    h, s = fill_subspace_matrices_from_fname_dict(fnames, ham_mpo, d)
    np.save("data/lih_h", h)
    np.save("data/lih_s", s)

    ds, energies, num_kept = energy_vs_d(h, s, eps)
    for dd, ener in zip(ds, energies):
        print(f"{dd} {ener}")
    
    # df = pd.DataFrame({"d": ds, "energy": energies})
    # df.set_index("d", inplace=True)
    # df.index.name = "d"
    # df.to_csv(args.output_file)

    f = h5py.File(args.output_file, "w")
    f.create_dataset("eps", data=eps)
    f.create_dataset("tau", data=tau)
    f.create_dataset("max_mpo_bond", data=max_mpo_bond)
    f.create_dataset("max_tebd_bond", data=max_tebd_bond)
    f.create_dataset("steps", data=steps)
    f.create_dataset("h", data=h)
    f.create_dataset("s", data=s)
    f.create_dataset("ds", data=np.array(ds))
    f.create_dataset("energies", data=np.array(energies))
    f.create_dataset("num_kept", data=np.array(num_kept))
    f.close()

if __name__ == "__main__":
    main()