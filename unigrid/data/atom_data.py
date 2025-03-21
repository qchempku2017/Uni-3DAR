atom_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

protein_atom_list = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]

protein_res_dict = {
    "ALA": ["C", "CA", "CB", "N", "O", "OXT"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2", "OXT"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2", "OXT"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1", "OXT"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG", "OXT"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2", "OXT"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1", "OXT"],
    "GLY": ["C", "CA", "N", "O", "OXT"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O", "OXT"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O", "OXT"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O", "OXT"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O", "OXT"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD", "OXT"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OXT"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O", "OXT"],
    "SER": ["C", "CA", "CB", "N", "O", "OG", "OXT"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1", "OXT"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
        "OXT",
    ],
    "TYR": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "N",
        "O",
        "OH",
        "OXT",
    ],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O", "OXT"],
    "X": protein_atom_list,
}

atom_additional_feat = {
    "H": ["1", "(-1, 1)", "1", "1"],
    "He": ["8", "()", "18", "1"],
    "Li": ["1", "(1,)", "1", "2"],
    "Be": ["2", "(2,)", "2", "2"],
    "B": ["3", "(3,)", "13", "2"],
    "C": ["4", "(-4, -3, -2, -1, 0, 1, 2, 3, 4)", "14", "2"],
    "N": ["5", "(-3, 3, 5)", "15", "2"],
    "O": ["6", "(-2,)", "16", "2"],
    "F": ["7", "(-1,)", "17", "2"],
    "Ne": ["8", "()", "18", "2"],
    "Na": ["1", "(1,)", "1", "3"],
    "Mg": ["2", "(2,)", "2", "3"],
    "Al": ["3", "(3,)", "13", "3"],
    "Si": ["4", "(-4, 4)", "14", "3"],
    "P": ["5", "(-3, 3, 5)", "15", "3"],
    "S": ["6", "(-2, 2, 4, 6)", "16", "3"],
    "Cl": ["7", "(-1, 1, 3, 5, 7)", "17", "3"],
    "Ar": ["8", "(0,)", "18", "3"],
    "K": ["1", "(1,)", "1", "4"],
    "Ca": ["2", "(2,)", "2", "4"],
    "Sc": ["2", "(3,)", "3", "4"],
    "Ti": ["2", "(2, 3, 4)", "4", "4"],
    "V": ["2", "(2, 3, 4, 5)", "5", "4"],
    "Cr": ["2", "(2, 3, 6)", "6", "4"],
    "Mn": ["2", "(2, 3, 4, 6, 7)", "7", "4"],
    "Fe": ["2", "(2, 3)", "8", "4"],
    "Co": ["2", "(2, 3)", "9", "4"],
    "Ni": ["2", "(2,)", "10", "4"],
    "Cu": ["2", "(1, 2)", "11", "4"],
    "Zn": ["2", "(2,)", "12", "4"],
    "Ga": ["3", "(3,)", "13", "4"],
    "Ge": ["4", "(-4, 2, 4)", "14", "4"],
    "As": ["5", "(-3, 3, 5)", "15", "4"],
    "Se": ["6", "(-2, 2, 4, 6)", "16", "4"],
    "Br": ["7", "(-1, 1, 3, 5)", "17", "4"],
    "Kr": ["8", "(0,)", "18", "4"],
    "Rb": ["1", "(1,)", "1", "5"],
    "Sr": ["2", "(2,)", "2", "5"],
    "Y": ["2", "(3,)", "3", "5"],
    "Zr": ["2", "(4,)", "4", "5"],
    "Nb": ["2", "(5,)", "5", "5"],
    "Mo": ["2", "(4, 6)", "6", "5"],
    "Tc": ["2", "(4, 7)", "7", "5"],
    "Ru": ["2", "(3, 4)", "8", "5"],
    "Rh": ["2", "(3,)", "9", "5"],
    "Pd": ["1", "(0, 2, 4)", "10", "5"],
    "Ag": ["2", "(1,)", "11", "5"],
    "Cd": ["2", "(2,)", "12", "5"],
    "In": ["3", "(3,)", "13", "5"],
    "Sn": ["4", "(-4, 2, 4)", "14", "5"],
    "Sb": ["5", "(-3, 3, 5)", "15", "5"],
    "Te": ["6", "(-2, 2, 4, 6)", "16", "5"],
    "I": ["7", "(-1, 1, 3, 5, 7)", "17", "5"],
    "Xe": ["8", "(0,)", "18", "5"],
    "Cs": ["1", "(1,)", "1", "6"],
    "Ba": ["2", "(2,)", "2", "6"],
    "La": ["2", "(3,)", "3", "6"],
    "Ce": ["3", "(3, 4)", "None", "6"],
    "Pr": ["2", "(3,)", "None", "6"],
    "Nd": ["2", "(3,)", "None", "6"],
    "Pm": ["2", "(3,)", "None", "6"],
    "Sm": ["2", "(3,)", "None", "6"],
    "Eu": ["2", "(2, 3)", "None", "6"],
    "Gd": ["3", "(3,)", "None", "6"],
    "Tb": ["2", "(3,)", "None", "6"],
    "Dy": ["2", "(3,)", "None", "6"],
    "Ho": ["2", "(3,)", "None", "6"],
    "Er": ["2", "(3,)", "None", "6"],
    "Tm": ["2", "(3,)", "None", "6"],
    "Yb": ["2", "(3,)", "None", "6"],
    "Lu": ["3", "(3,)", "None", "6"],
    "Hf": ["2", "(4,)", "4", "6"],
    "Ta": ["2", "(5,)", "5", "6"],
    "W": ["2", "(4, 6)", "6", "6"],
    "Re": ["2", "(4, 7)", "7", "6"],
    "Os": ["2", "(4,)", "8", "6"],
    "Ir": ["2", "(3, 4)", "9", "6"],
    "Pt": ["2", "(2, 4)", "10", "6"],
    "Au": ["2", "(1, 3)", "11", "6"],
    "Hg": ["2", "(1, 2)", "12", "6"],
    "Tl": ["3", "(1, 3)", "13", "6"],
    "Pb": ["4", "(2, 4)", "14", "6"],
    "Bi": ["5", "(3,)", "15", "6"],
    "Po": ["6", "(-2, 2, 4)", "16", "6"],
    "At": ["7", "(-1, 1)", "17", "6"],
    "Rn": ["8", "(2,)", "18", "6"],
    "Fr": ["1", "(1,)", "1", "7"],
    "Ra": ["2", "(2,)", "2", "7"],
    "Ac": ["2", "(3,)", "3", "7"],
    "Th": ["None", "(4,)", "None", "7"],
    "Pa": ["3", "(5,)", "None", "7"],
    "U": ["3", "(4, 6)", "None", "7"],
    "Np": ["3", "(5,)", "None", "7"],
    "Pu": ["2", "(4,)", "None", "7"],
    "Am": ["2", "(3,)", "None", "7"],
    "Cm": ["3", "(3,)", "None", "7"],
    "Bk": ["2", "(3,)", "None", "7"],
    "Cf": ["2", "(3,)", "None", "7"],
    "Es": ["2", "(3,)", "None", "7"],
    "Fm": ["2", "(3,)", "None", "7"],
    "Md": ["2", "(3,)", "None", "7"],
    "No": ["2", "(2,)", "None", "7"],
    "Lr": ["3", "(3,)", "None", "7"],
    "Rf": ["2", "(4,)", "4", "7"],
    "Db": ["2", "(5,)", "5", "7"],
    "Sg": ["2", "(6,)", "6", "7"],
    "Bh": ["2", "(7,)", "7", "7"],
    "Hs": ["2", "(8,)", "8", "7"],
    "Mt": ["2", "()", "9", "7"],
    "Ds": ["2", "()", "10", "7"],
    "Rg": ["2", "()", "11", "7"],
    "Cn": ["2", "(2,)", "12", "7"],
    "Nh": ["3", "()", "13", "7"],
    "Fl": ["4", "()", "14", "7"],
    "Mc": ["5", "()", "15", "7"],
    "Lv": ["6", "()", "16", "7"],
    "Ts": ["7", "()", "17", "7"],
    "Og": ["8", "()", "18", "7"],
}
