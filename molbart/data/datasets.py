import random
import functools
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from rdkit import Chem
from typing import Optional
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor


class _AbsDataset(Dataset):
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    def split_idxs(self, val_idxs, test_idxs):
        raise NotImplementedError()

    def split(self, val_perc=0.2, test_perc=0.2):
        """ Split the dataset randomly into three datasets

        Splits the dataset into train, validation and test datasets.
        Validation and test dataset have round(len * <val/test>_perc) elements in each
        """

        split_perc = val_perc + test_perc
        if split_perc > 1:
            msg = f"Percentage of dataset to split must not be greater than 1, got {split_perc}"
            raise ValueError(msg)

        dataset_len = len(self)
        val_len = round(dataset_len * val_perc)
        test_len = round(dataset_len * test_perc)

        val_idxs = random.sample(range(dataset_len), val_len)
        test_idxs = random.sample(range(dataset_len), test_len)

        train_dataset, val_dataset, test_dataset = self.split_idxs(val_idxs, test_idxs)

        return train_dataset, val_dataset, test_dataset


class ReactionDataset(_AbsDataset):
    def __init__(self, reactants, products, seq_lengths=None):
        super(ReactionDataset, self).__init__()

        if len(reactants) != len(products):
            raise ValueError(f"There must be an equal number of reactants and products")

        self.reactants = reactants
        self.products = products
        self.seq_lengths = seq_lengths

    def __len__(self):
        return len(self.reactants)

    def __getitem__(self, item):
        reactant_mol = self.reactants[item]
        product_mol = self.products[item]
        return reactant_mol, product_mol

    def split_idxs(self, val_idxs, test_idxs):
        """ Splits dataset into train, val and test

        Note: Assumes all remaining indices outside of val_idxs and test_idxs are for training data
        The datasets are returned as ReactionDataset objects, if these should be a subclass 
        the from_reaction_pairs function should be overidden

        Args:
            val_idxs (List[int]): Indices for validation data
            test_idxs (List[int]): Indices for test data

        Returns:
            (ReactionDataset, ReactionDataset, ReactionDataset): Train, val and test datasets
        """

        val_data = [self[idx] for idx in val_idxs]
        val_lengths = [self.seq_lengths[idx] for idx in val_idxs] if self.seq_lengths is not None else None
        val_dataset = self.from_reaction_pairs(val_data, lengths=val_lengths)

        test_data = [self[idx] for idx in test_idxs]
        test_lengths = [self.seq_lengths[idx] for idx in test_idxs] if self.seq_lengths is not None else None
        test_dataset = self.from_reaction_pairs(test_data, lengths=test_lengths)

        train_idxs = set(range(len(self))) - set(val_idxs).union(set(test_idxs))
        train_data = [self[idx] for idx in sorted(train_idxs)]
        train_lengths = [self.seq_lengths[idx] for idx in train_idxs] if self.seq_lengths is not None else None
        train_dataset = self.from_reaction_pairs(train_data, lengths=train_lengths)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def from_reaction_pairs(reaction_pairs, lengths=None):
        reacts, prods = tuple(zip(*reaction_pairs))
        dataset = ReactionDataset(reacts, prods, seq_lengths=lengths)
        return dataset


class Uspto50(ReactionDataset):
    def __init__(self, data_path):
        path = Path(data_path)
        df = pd.read_pickle(path)
        reactants = df["reactant_ROMol"].tolist()
        products = df["products_ROMol"].tolist()
        
        super().__init__(reactants, products)

        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _save_idxs(self, df):
        val_idxs = df.index[df["set"] == "valid"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class UsptoMit(ReactionDataset):
    def __init__(self, data_path):
        path = Path(data_path)
        df = pd.read_pickle(path)
        reactants = df["reactants_mol"].tolist()
        products = df["products_mol"].tolist()
        reactant_lengths = df["reactant_lengths"].tolist()
        product_lengths = df["product_lengths"].tolist()

        super().__init__(reactants, products, seq_lengths=product_lengths)

        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _save_idxs(self, df):
        val_idxs = df.index[df["set"] == "valid"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class MolOptDataset(ReactionDataset):
    def __init__(self, data_path):
        path = Path(data_path)
        df = pd.read_csv(path)
        data_in = df["Input"].tolist()
        data_out = df["Output"].tolist()

        super().__init__(data_in, data_out)

        self.train_idxs, self.val_idxs, self.test_idxs = self._save_idxs(df)

    def _save_idxs(self, df):
        val_idxs = df.index[df["Set"] == "validation"].tolist()
        test_idxs = df.index[df["Set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class MoleculeDataset(_AbsDataset):
    def __init__(
        self,
        molecules,
        seq_lengths=None,
        transform=None,
        train_idxs=None,
        val_idxs=None,
        test_idxs=None
    ):
        super(MoleculeDataset, self).__init__()

        self.molecules = molecules
        self.seq_lengths = seq_lengths
        self.transform = transform
        self.train_idxs = train_idxs
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, item):
        molecule = self.molecules[item]
        if self.transform is not None:
            molecule = self.transform(molecule)

        return molecule

    def split_idxs(self, val_idxs, test_idxs):
        val_mols = [self.molecules[idx] for idx in val_idxs]
        val_lengths = [self.seq_lengths[idx] for idx in val_idxs] if self.seq_lengths is not None else None
        val_dataset = MoleculeDataset(val_mols, val_lengths, self.transform)

        test_mols = [self.molecules[idx] for idx in test_idxs]
        test_lengths = [self.seq_lengths[idx] for idx in test_idxs] if self.seq_lengths is not None else None
        test_dataset = MoleculeDataset(test_mols, test_lengths, self.transform)

        train_idxs = set(range(len(self))) - set(val_idxs).union(set(test_idxs))
        train_mols = [self.molecules[idx] for idx in sorted(train_idxs)]
        train_lengths = [self.seq_lengths[idx] for idx in train_idxs] if self.seq_lengths is not None else None
        train_dataset = MoleculeDataset(train_mols, train_lengths, self.transform)

        return train_dataset, val_dataset, test_dataset


class Chembl(MoleculeDataset):
    def __init__(self, data_path):
        path = Path(data_path)
        df = pd.read_pickle(path)

        molecules = df["molecules"].tolist()
        lengths = df["lengths"].tolist()
        train_idxs, val_idxs, test_idxs = self._save_idxs(df)

        super().__init__(
            molecules,
            seq_lengths=lengths,
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs
        )

    def _save_idxs(self, df):
        val_idxs = df.index[df["set"] == "val"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class ZincSlice(MoleculeDataset):
    def __init__(self, df):
        smiles = df["smiles"].tolist()
        train_idxs, val_idxs, test_idxs = self._save_idxs(df)

        super().__init__(
            smiles,
            train_idxs=train_idxs,
            val_idxs=val_idxs,
            test_idxs=test_idxs,
            transform=lambda smi: Chem.MolFromSmiles(smi)
        )

    def _save_idxs(self, df):
        val_idxs = df.index[df["set"] == "val"].tolist()
        test_idxs = df.index[df["set"] == "test"].tolist()

        idxs_intersect = set(val_idxs).intersection(set(test_idxs))
        if len(idxs_intersect) > 0:
            raise ValueError(f"Val idxs and test idxs overlap")

        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))

        return train_idxs, val_idxs, test_idxs


class Zinc(ZincSlice):
    def __init__(self, data_path):
        path = Path(data_path)

        # If path is a directory then read every subfile
        if path.is_dir():
            df = self._read_dir_df(path)
        else:
            df = pd.read_csv(path)

        super().__init__(df)

    def _read_dir_df(self, path):
        # num_cpus = 4
        # executor = ProcessPoolExecutor(num_cpus)
        # files = [f for f in path.iterdir()]
        # futures = [executor.submit(pd.read_csv, f) for f in files]
        # dfs = [future.result() for future in futures]

        dfs = [pd.read_csv(f) for f in path.iterdir()]

        zinc_df = pd.concat(dfs, ignore_index=True, copy=False)
        return zinc_df


class ConcatMoleculeDataset(MoleculeDataset):
    """ Dataset class for storing (concatenated) molecules 

    Automatically constructs a dataset which contains rdkit molecules
    Roughly a third of these molecule objects are single molecules,
    another third contain two molecules and the final third contain three molecules.

    The molecules to be concatenated are randomly selected, 
    so the ordering from the original data is not preserved.
    """

    def __init__(
        self, 
        dataset: MoleculeDataset,
        join_token: Optional[str] = ".",
        double_mol_prob: Optional[float] = 0.333,
        triple_mol_prob: Optional[float] = 0.333
    ):
        self.join_token = join_token
        self.double_mol_prob = double_mol_prob
        self.triple_mol_prob = triple_mol_prob

        self.original_dataset = dataset

        concat_idxs = self._construct_concat_idxs(dataset)

        super(ConcatMoleculeDataset, self).__init__(
            concat_idxs, 
            transform=self._process_molecule_idxs,
            train_idxs=dataset.train_idxs,
            val_idxs=dataset.val_idxs,
            test_idxs=dataset.test_idxs
        )

    def _construct_concat_idxs(self, dataset):
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)

        curr = 0
        molecule_idxs = []

        added_prob = self.double_mol_prob + self.triple_mol_prob
        
        while curr <= len(idxs) - 1:
            rand = random.random()

            # Use two molecules
            if rand < self.double_mol_prob and curr <= len(idxs) - 2:
                curr_idxs = [idxs[curr + i] for i in range(2)]
                molecule_idxs.append(curr_idxs)
                curr += 2

            # Or, Use three molecules together
            elif rand < added_prob and curr <= len(idxs) - 3:
                curr_idxs = [idxs[curr + i] for i in range(3)]
                molecule_idxs.append(curr_idxs)
                curr += 3

            # Or, take a single molecule
            else:
                curr_idx = idxs[curr]
                molecule_idxs.append([curr_idx])
                curr += 1

        return molecule_idxs

    def _process_molecule_idxs(self, idxs):
        if len(idxs) == 1:
            molecule = self.original_dataset[idxs[0]]
        else:
            molecule = self._concat_mols_from_idxs(idxs, self.original_dataset)

        return molecule

    def _concat_mols_from_idxs(self, idxs, dataset):
        mols = [dataset[idx] for idx in idxs]
        concat_mol = functools.reduce(lambda m1, m2: Chem.CombineMols(m1, m2), mols)
        return concat_mol
