import os
import sys
import contextlib
import tempfile
from rdkit import Chem, RDLogger
from absl import logging
from tqdm import tqdm
import json
import time


from bond_analyze import read_xyz_file, build_xae_molecule
from multiprocessing import Pool
from functools import partial

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


@contextlib.contextmanager
def supress_stdout():
    in_memory_file = tempfile.SpooledTemporaryFile()
    # suppress stdout
    orig_stdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(in_memory_file.fileno(), 1)
    # suppress stderr
    orig_stderr_fno = os.dup(sys.stderr.fileno())
    os.dup2(in_memory_file.fileno(), 2)
    try:
        yield
    finally:
        os.fsync(in_memory_file)
        os.dup2(orig_stdout_fno, 1)  # restore stdout
        os.dup2(orig_stderr_fno, 2)  # restore stderr
        in_memory_file.seek(0)
        outputs = in_memory_file.read().decode("utf-8").strip()
        if outputs:
            logging.info(outputs)
        in_memory_file.close()


def mol2smiles(mol):
    with supress_stdout():
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return Chem.MolToSmiles(mol, canonical=True)


def compute_validity(mol):
    if mol is None:
        return 0, None
    if mol2smiles(mol) is None:
        return 0, None
    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    smiles = mol2smiles(largest_mol)
    if smiles is not None:
        return 1, smiles
    else:
        return 0, None


def compute_uniqueness(valid):
    return list(set(valid)), len(set(valid)) / (len(valid) + 1e-12)


def worker(data, single_bond=False):
    positions, atom_types = data
    mol, molecule_stable_i, nr_stable_bonds_i, num_atoms = build_xae_molecule(
        positions, atom_types, single_bond=single_bond
    )
    valid_, smiles = compute_validity(mol)
    return molecule_stable_i, nr_stable_bonds_i, num_atoms, valid_, smiles


def get_metric(results):
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    valid = 0
    valid_list = []
    for i in tqdm(results):
        molecule_stable_i, nr_stable_bonds_i, num_atoms, valid_, smiles = i
        molecule_stable += int(molecule_stable_i)
        nr_stable_bonds += int(nr_stable_bonds_i)
        n_atoms += int(num_atoms)

        valid += valid_
        if smiles is not None:
            valid_list.append(smiles)
    return molecule_stable, nr_stable_bonds, n_atoms, valid, valid_list


def get_stable_valid_unique_novel(xyz_data, dataset="qm9"):
    all_files_len = len(xyz_data)

    with Pool() as pool:
        results = list(tqdm(pool.imap(worker, xyz_data), total=len(xyz_data)))
    molecule_stable, nr_stable_bonds, n_atoms, valid, valid_list = get_metric(results)
    if dataset == "drug":
        with Pool() as pool:
            worker_func = partial(worker, single_bond=True)
            results = list(tqdm(pool.imap(worker_func, xyz_data), total=len(xyz_data)))
        _, _, _, valid, valid_list = get_metric(results)
    fraction_mol_stable = molecule_stable / all_files_len
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    fraction_valid = valid / all_files_len
    unique = list(set(valid_list))
    fraction_unique = len(unique) / (len(valid_list) + 1e-12)
    result = {
        "mol_stable": fraction_mol_stable,
        "atom_stable": fraction_atm_stable,
        "valid": fraction_valid,
        "unique": fraction_unique,
    }
    return result  # , valid, unique, novel


def main(input_file, output_file, dataset="qm9"):
    result_dict = {}

    pred_xyz_data = read_xyz_file(input_file)
    result_dict_ = get_stable_valid_unique_novel(pred_xyz_data, dataset)
    result_dict.update(result_dict_)
    result_dict["num_samples"] = len(pred_xyz_data)
    print(result_dict)
    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=4)
    return result_dict


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) >= 4:
        dataset = sys.argv[3]
    else:
        dataset = "qm9"
    start = time.time()
    main(input_file, output_file, dataset)
    end = time.time()
    print("Time taken: ", end - start)
