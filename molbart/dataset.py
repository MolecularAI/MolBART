import random
import functools
import multiprocessing
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from rdkit import Chem
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler
from pysmilesutils.augment import MolRandomizer

from molenc.tokenise import MolEncTokeniser


# --------------------------------------------------------------------------------------------------------
# -------------------------------------------- Datasets --------------------------------------------------
# --------------------------------------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Data Modules ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class _AbsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        tokeniser,
        batch_size,
        max_seq_len,
        train_token_batch_size=None,
        num_buckets=None,
        val_idxs=None, 
        test_idxs=None,
        split_perc=0.2
    ):
        super(_AbsDataModule, self).__init__()

        if val_idxs is not None and test_idxs is not None:
            idxs_intersect = set(val_idxs).intersection(set(test_idxs))
            if len(idxs_intersect) > 0:
                raise ValueError(f"Val idxs and test idxs overlap")

        if train_token_batch_size is not None and num_buckets is not None:
            print(f"""Training with approx. {train_token_batch_size} tokens per batch"""
                f""" and {num_buckets} buckets in the sampler.""")
        else:
            print(f"Using a batch size of {batch_size} for training.")

        self.dataset = dataset
        self.tokeniser = tokeniser

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_token_batch_size = train_token_batch_size
        self.num_buckets = num_buckets
        self.val_idxs = val_idxs
        self.test_idxs = test_idxs
        self.split_perc = split_perc

        self._num_workers = multiprocessing.cpu_count()

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # Use train_token_batch_size with TokenSampler for training and batch_size for validation and testing
    def train_dataloader(self):
        if self.train_token_batch_size is None:
            loader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size,
                num_workers=self._num_workers, 
                collate_fn=self._collate,
                shuffle=True
            )
            return loader

        sampler = TokenSampler(
            self.num_buckets,
            self.train_dataset.seq_lengths,
            self.train_token_batch_size,
            shuffle=True
        )
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=self._collate
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self._num_workers, 
            collate_fn=self._collate
        )
        return loader

    def setup(self, stage=None):
        train_dataset = None
        val_dataset = None
        test_dataset = None

        # Split datasets by idxs passed in...
        if self.val_idxs is not None and self.test_idxs is not None:
            train_dataset, val_dataset, test_dataset = self.dataset.split_idxs(self.val_idxs, self.test_idxs)

        # Or randomly
        else:
            train_dataset, val_dataset, test_dataset = self.dataset.split(self.split_perc, self.split_perc)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def _collate(self, batch):
        raise NotImplementedError()

    def _check_seq_len(self, tokens, mask):
        """ Warn user and shorten sequence if the tokens are too long, otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened, if necessary)
            mask (List[List[int]]): List of mask sequences (shortened, if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.max_seq_len:
            print(f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size")

            tokens_short = [ts[:self.max_seq_len] for ts in tokens]
            mask_short = [ms[:self.max_seq_len] for ms in mask]

            return tokens_short, mask_short

        return tokens, mask


class MoleculeDataModule(_AbsDataModule):
    def __init__(
        self,
        dataset: MoleculeDataset,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        train_token_batch_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        val_idxs: Optional[List[int]] = None, 
        test_idxs: Optional[List[int]] = None,
        split_perc: Optional[float] = 0.2,
        augment: Optional[bool] = True 
    ):
        super(MoleculeDataModule, self).__init__(
            dataset,
            tokeniser,
            batch_size,
            max_seq_len,
            train_token_batch_size,
            num_buckets,
            val_idxs, 
            test_idxs,
            split_perc
        )

        self.aug = MolRandomizer() if augment else None

    def _collate(self, batch):
        token_output = self._prepare_tokens(batch)
        enc_tokens = token_output["encoder_tokens"]
        enc_pad_mask = token_output["encoder_pad_mask"]
        dec_tokens = token_output["decoder_tokens"]
        dec_pad_mask = token_output["decoder_pad_mask"]
        target_smiles = token_output["target_smiles"]

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.tokeniser.convert_tokens_to_ids(dec_tokens)

        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_pad_mask = torch.tensor(enc_pad_mask, dtype=torch.bool).transpose(0, 1)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
        dec_pad_mask = torch.tensor(dec_pad_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids[:-1, :],
            "decoder_pad_mask": dec_pad_mask[:-1, :],
            "target": dec_token_ids.clone()[1:, :],
            "target_pad_mask": dec_pad_mask.clone()[1:, :],
            "target_smiles": target_smiles
        }

        return collate_output

    def _prepare_tokens(self, batch):
        if self.aug is None:
            encoder_mols = batch
        else:
            encoder_mols = self.aug(batch)

        if self.aug is None:
            decoder_mols = batch
        else:
            decoder_mols = self.aug(batch)

        canonical = self.aug is None
        enc_smiles = [Chem.MolToSmiles(mol, canonical=canonical) for mol in encoder_mols]
        dec_smiles = [Chem.MolToSmiles(mol, canonical=canonical) for mol in decoder_mols]

        enc_token_output = self.tokeniser.tokenise(enc_smiles, mask=True, pad=True)
        dec_token_output = self.tokeniser.tokenise(dec_smiles, pad=True)

        enc_tokens = enc_token_output["masked_tokens"]
        enc_mask = enc_token_output["pad_masks"]
        dec_tokens = dec_token_output["original_tokens"]
        dec_mask = dec_token_output["pad_masks"]

        enc_tokens, enc_mask = self._check_seq_len(enc_tokens, enc_mask)
        dec_tokens, dec_mask = self._check_seq_len(dec_tokens, dec_mask)

        token_output = {
            "encoder_tokens": enc_tokens,
            "encoder_pad_mask": enc_mask,
            "decoder_tokens": dec_tokens,
            "decoder_pad_mask": dec_mask,
            "target_smiles": dec_smiles
        }

        return token_output


class FineTuneReactionDataModule(_AbsDataModule):
    def __init__(
        self,
        dataset: ReactionDataset,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        train_token_batch_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        forward_pred: Optional[bool] = True,
        val_idxs: Optional[List[int]] = None, 
        test_idxs: Optional[List[int]] = None,
        split_perc: Optional[float] = 0.2,
        augment: Optional[str] = None
    ):
        super().__init__(
            dataset,
            tokeniser,
            batch_size,
            max_seq_len,
            train_token_batch_size,
            num_buckets,
            val_idxs, 
            test_idxs,
            split_perc
        )

        if augment is None:
            print("No data augmentation.")
        elif augment == "reactants":
            print("Augmenting reactants only.")
        elif augment == "all":
            print("Augmenting both reactants and products.")
        else:
            raise ValueError(f"Unknown value for augment, {augment}")

        self.augment = augment
        self.aug = MolRandomizer() if augment is not None else None
        self.forward_pred = forward_pred

    def _collate(self, batch):
        # TODO Allow both forward and backward prediction

        token_output = self._prepare_tokens(batch)
        reacts_tokens = token_output["reacts_tokens"]
        reacts_mask = token_output["reacts_mask"]
        prods_tokens = token_output["prods_tokens"]
        prods_mask = token_output["prods_mask"]
        prods_smiles = token_output["products_smiles"]

        reacts_token_ids = self.tokeniser.convert_tokens_to_ids(reacts_tokens)
        prods_token_ids = self.tokeniser.convert_tokens_to_ids(prods_tokens)

        reacts_token_ids = torch.tensor(reacts_token_ids).transpose(0, 1)
        reacts_pad_mask = torch.tensor(reacts_mask, dtype=torch.bool).transpose(0, 1)
        prods_token_ids = torch.tensor(prods_token_ids).transpose(0, 1)
        prods_pad_mask = torch.tensor(prods_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "encoder_input": reacts_token_ids,
            "encoder_pad_mask": reacts_pad_mask,
            "decoder_input": prods_token_ids[:-1, :],
            "decoder_pad_mask": prods_pad_mask[:-1, :],
            "target": prods_token_ids.clone()[1:, :],
            "target_pad_mask": prods_pad_mask.clone()[1:, :],
            "target_smiles": prods_smiles
        }

        return collate_output

    def _prepare_tokens(self, batch):
        """ Prepare smiles strings for the model

        RDKit is used to construct a smiles string
        The smiles strings are prepared for the forward prediction task, no masking

        Args:
            batch (list[tuple(Chem.Mol, Chem.Mol)]): Batched input to the model

        Output:
            Dictionary output from tokeniser: {
                "reacts_tokens" (List[List[str]]): Reactant tokens from tokeniser,
                "prods_tokens" (List[List[str]]): Product tokens from tokeniser,
                "reacts_masks" (List[List[int]]): 0 refers to not padded, 1 refers to padded,
                "prods_masks" (List[List[int]]): 0 refers to not padded, 1 refers to padded
            }
        """

        reacts, prods = tuple(zip(*batch))

        if self.augment == "reactants" or self.augment == "all":
            reacts = [Chem.MolToSmiles(react, canonical=False) for react in self.aug(reacts)]
        else:
            reacts = [Chem.MolToSmiles(react) for react in reacts]

        if self.augment == "all":
            prods = [Chem.MolToSmiles(prod, canonical=False) for prod in self.aug(prods)]
        else:
            prods = [Chem.MolToSmiles(prod) for prod in prods]

        reacts_output = self.tokeniser.tokenise(reacts, pad=True)
        prods_output = self.tokeniser.tokenise(prods, pad=True)

        reacts_tokens = reacts_output["original_tokens"]
        reacts_mask = reacts_output["pad_masks"]
        reacts_tokens, reacts_mask = self._check_seq_len(reacts_tokens, reacts_mask)

        prods_tokens = prods_output["original_tokens"]
        prods_mask = prods_output["pad_masks"]
        prods_tokens, prods_mask = self._check_seq_len(prods_tokens, prods_mask)

        token_output = {
            "reacts_tokens": reacts_tokens,
            "reacts_mask": reacts_mask,
            "prods_tokens": prods_tokens,
            "prods_mask": prods_mask,
            "reactants_smiles": reacts,
            "products_smiles": prods
        }

        return token_output


class FineTuneMolOptDataModule(_AbsDataModule):
    def __init__(
        self,
        dataset: ReactionDataset,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        val_idxs: Optional[List[int]] = None, 
        test_idxs: Optional[List[int]] = None,
        split_perc: Optional[float] = 0.2
    ):
        super().__init__(
            dataset,
            tokeniser,
            batch_size,
            max_seq_len,
            val_idxs, 
            test_idxs,
            split_perc
        )

    def _collate(self, batch):
        token_output = self._prepare_tokens(batch)
        reacts_tokens = token_output["reacts_tokens"]
        reacts_mask = token_output["reacts_mask"]
        prods_tokens = token_output["prods_tokens"]
        prods_mask = token_output["prods_mask"]
        prods_smiles = token_output["products_smiles"]

        reacts_token_ids = self.tokeniser.convert_tokens_to_ids(reacts_tokens)
        prods_token_ids = self.tokeniser.convert_tokens_to_ids(prods_tokens)

        reacts_token_ids = torch.tensor(reacts_token_ids).transpose(0, 1)
        reacts_pad_mask = torch.tensor(reacts_mask, dtype=torch.bool).transpose(0, 1)
        prods_token_ids = torch.tensor(prods_token_ids).transpose(0, 1)
        prods_pad_mask = torch.tensor(prods_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            "encoder_input": reacts_token_ids,
            "encoder_pad_mask": reacts_pad_mask,
            "decoder_input": prods_token_ids[:-1, :],
            "decoder_pad_mask": prods_pad_mask[:-1, :],
            "target": prods_token_ids.clone()[1:, :],
            "target_pad_mask": prods_pad_mask.clone()[1:, :],
            "target_smiles": prods_smiles
        }

        return collate_output

    def _prepare_tokens(self, batch):
        # TODO

        pass


# ----------------------------------------------------------------------------------------------------------
# --------------------------------------------- Helper Classes ---------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class TokenSampler(Sampler):
    """
    A Sampler which groups sequences into buckets based on length and constructs batches using 
    a (potentially) different number of sequences from each bucket to achieve a target number of 
    tokens in each batch. This approach has a number of advantages:
        - Faster training and eval since there are fewer pad tokens vs random batching
        - Potentially improved training stability since the number of tokens is approx the same
          each batch

    Note: There is a systematic error in the batch size (it will be slightly larger than the 
          target size on average) since we simply take the mean of the seq lengths in the bucket,
          this does not account for padding that will result from the largest seq in the batch.
    """

    def __init__(
        self,
        num_buckets,
        seq_lengths,
        batch_size,
        shuffle=True,
        drop_last=True
    ):
        """ Init method

        Args:
            num_buckets (int): Number of buckets to split sequences into
            seq_lengths (List[int]): The length of the sequences in the dataset (in the same order)
            batch_size (int): Target number of tokens in each batch
            shuffle (Optional[bool]): Shuffle the indices within each bucket
            drop_last (Optional[bool]): Forget about the indices remaining at the end of each bucket
        """

        if not drop_last:
            raise NotImplementedError("Keeping last elements is not yet supported")

        min_length = min(seq_lengths)
        max_length = max(seq_lengths) + 1
        bucket_width = (max_length - min_length) / num_buckets

        bucket_limits = []
        lower_limit = float(min_length)

        # Setup lower (inclusive) and upper (exclusive) seq length limits on buckets
        for _ in range(num_buckets):
            upper_limit = lower_limit + bucket_width
            bucket_limits.append((lower_limit, upper_limit))
            lower_limit = upper_limit

        buckets = [[] for _ in range(num_buckets)]
        lengths = [[] for _ in range(num_buckets)]

        # Add indices to correct bucket based on seq length
        for seq_idx, length in enumerate(seq_lengths):
            for b_idx, (lower, upper) in enumerate(bucket_limits):
                if lower <= length < upper:
                    buckets[b_idx].append(seq_idx)
                    lengths[b_idx].append(length)

        if shuffle:
            samplers = [RandomSampler(idxs) for idxs in buckets]
        else:
            samplers = [SequentialSampler(idxs) for idxs in buckets]

        # Work out approx number of sequences required for each bucket
        avg_lengths = [sum(ls) // len(ls) for ls in lengths]
        num_seqs = [batch_size // length for length in avg_lengths]
        num_seqs = [int(num_sq) for num_sq in num_seqs]

        num_batches = [len(bucket) // num_seqs[b_idx] for b_idx, bucket in enumerate(buckets)]
        num_batches = [int(num_bs) for num_bs in num_batches]

        self.num_seqs = num_seqs
        self.buckets = buckets
        self.num_batches = num_batches
        self.samplers = samplers

    def __iter__(self):
        iters = [iter(sampler) for sampler in self.samplers]
        rem_batches = self.num_batches[:]
        while sum(rem_batches) > 0:
            b_idx = random.choices(range(len(rem_batches)), weights=rem_batches, k=1)[0]
            batch_idxs = [next(iters[b_idx]) for _ in range(self.num_seqs[b_idx])]
            batch = [self.buckets[b_idx][idx] for idx in batch_idxs]
            rem_batches[b_idx] -= 1
            yield batch

    def __len__(self):
        return sum(self.num_batches)
