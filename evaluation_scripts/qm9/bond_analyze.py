# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf

import numpy as np
from rdkit import Chem
from tqdm import tqdm


bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}

bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}


margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}


def get_bond_order(atom1, atom2, distance, check_exists=True, single_bond=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            print(f"Atom {atom1} not in bonds1")
            return 0
        if atom2 not in bonds1[atom1]:
            print(f"Atom {atom2} not in bonds1[{atom1}]")
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:
        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3 if not single_bond else 1  # Triple
                return 2 if not single_bond else 1  # Double
        return 1  # Single
    return 0  # No bond


def build_xae_molecule(
    positions, atom_type, single_bond=False, mol_wanted=True, save_path=None
):
    n = positions.shape[0]
    assert positions.shape[1] == 3, "positions 必须是 (n_atoms, 3) 的数组"
    assert len(atom_type) == n, "positions 和 atom_type 的长度不匹配"

    A = np.zeros((n, n), dtype=bool)
    E = np.zeros((n, n), dtype=int)

    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    for i in range(n):
        for j in range(i):
            order = get_bond_order(
                atom_type[i],
                atom_type[j],
                dists[i, j],
                single_bond=single_bond,
            )
            if order > 0:
                A[i, j] = A[j, i] = 1
                E[i, j] = E[j, i] = order
    if mol_wanted is False:
        return n, A, E, dists
    mol = Chem.RWMol()
    for atom_symbol in atom_type:
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)

    bond_dict = {
        0: None,
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        1.5: Chem.rdchem.BondType.AROMATIC,
    }
    nonzero_indices = np.nonzero(A)
    for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
        if i > j:
            bond_type = bond_dict.get(E[i, j], None)
            if bond_type is not None:
                mol.AddBond(int(i), int(j), bond_type)
            else:
                print(f"Warning: Invalid bond type {E[i, j]} at indices ({i}, {j})")

    conf = Chem.Conformer(n)
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, tuple(pos))
    mol.AddConformer(conf)
    nr_stable_bonds = 0
    nr_bonds = E.sum(1)
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        possible_bonds = allowed_bonds[atom_type_i]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == nr_bonds_i
        else:
            is_stable = nr_bonds_i in possible_bonds
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(atom_type)
    if save_path:
        Chem.MolToMolFile(mol, save_path)
    return mol, molecule_stable, nr_stable_bonds, len(atom_type)


def read_xyz_file_single(lines):
    atom_types = []
    positions = []
    num_atoms = int(lines[0].strip())
    value_socre = lines[1].replace("score:", "").strip().split(" ")
    value_socre = float(value_socre[-1])
    for line in lines[2 : 2 + num_atoms]:
        parts = line.split()
        atom_types.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    positions = np.array(positions)
    return positions, atom_types


def read_xyz_file(file_path):
    molecules = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    i = 0
    with tqdm(total=len(lines), desc="Processing XYZ file", unit="lines") as pbar:
        while i < len(lines):
            num_atoms = int(lines[i].strip())
            result = read_xyz_file_single(lines[i : i + num_atoms + 2])
            i += num_atoms + 2
            molecules.append(result)
            pbar.update(num_atoms + 2)
    return molecules
