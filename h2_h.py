import numpy as np
from scipy.sparse.linalg import norm
import pandas as pd
import openfermion as of
from openfermionpyscf import run_pyscf
from fermion_helpers import add_number_term, fock_state
from krylov import (
    generate_u_subspace,
    generate_h_subspace,
    fill_subspace_matrices_full,
    energy_vs_d
)

def main():
    molec = "LiH"
    basis = "sto-3g"
    n_elec = 4
    alpha = 1.0
    d = 60
    eps = 1e-8

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
    ham_sparse = of.linalg.get_sparse_operator(ham_augmented)
    tau = np.pi / norm(ham_sparse)

    ref_state = fock_state([True] * n_elec + [False] * (nq - n_elec))
    states = generate_h_subspace(ref_state, ham_sparse, d)
    h, s = fill_subspace_matrices_full(ham_sparse, states)
    ds, energies = energy_vs_d(h, s, eps)
    for dd, ener in zip(ds, energies):
        print(f"{dd} {ener}")
    
    df = pd.DataFrame({"d": ds, "energy": energies})
    df.set_index("d", inplace=True)
    df.index.name = "d"
    df.to_csv("data/h2_h.csv")

if __name__ == "__main__":
    main()