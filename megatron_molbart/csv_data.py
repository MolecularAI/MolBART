import torch
from torch.utils.data import Dataset
from pysmilesutils.augment import MolAugmenter, SMILESAugmenter
from pysmilesutils.datautils import BucketBatchSampler
from molbart.tokeniser import MolEncTokeniser
from molbart.util import DEFAULT_CHEM_TOKEN_START
from molbart.util import DEFAULT_VOCAB_PATH
from molbart.util import DEFAULT_MAX_SEQ_LEN
from molbart.util import REGEX
from rdkit import Chem
import numpy as np
import pandas
from molbart.data.util import TokenSampler
from megatron.data.samplers import DistributedBatchSampler
from megatron import mpu
import torch
from rdkit import Chem # ADDED FOR FINE-TUNING

tokenizer = MolEncTokeniser.from_vocab_file(DEFAULT_VOCAB_PATH, REGEX,
        DEFAULT_CHEM_TOKEN_START)
max_seq_len = 512


def check_seq_len(tokens, mask):
    """ Warn user and shorten sequence if the tokens are too long, otherwise return original

    Args:
        tokens (List[List[str]]): List of token sequences
        mask (List[List[int]]): List of mask sequences

    Returns:
        tokens (List[List[str]]): List of token sequences (shortened, if necessary)
        mask (List[List[int]]): List of mask sequences (shortened, if necessary)
    """

    seq_len = max([len(ts) for ts in tokens])
    if seq_len > max_seq_len:
        tokens_short = [ts[:max_seq_len] for ts in tokens]
        mask_short = [ms[:max_seq_len] for ms in mask]
        return (tokens_short, mask_short)
    return (tokens, mask)


def collate_fn(batch, forward=True):
    """ Used by DataLoader to concatenate/collate inputs."""

    encoder_smiles = [x['encoder_smiles'] for x in batch]
    decoder_smiles = [x['decoder_smiles'] for x in batch]

    enc_token_output = tokenizer.tokenise(encoder_smiles, mask=True,
            pad=True)
    dec_token_output = tokenizer.tokenise(decoder_smiles, pad=True)

    enc_mask = enc_token_output['masked_pad_masks']
    enc_tokens = enc_token_output['masked_tokens']
    dec_tokens = dec_token_output['original_tokens']
    dec_mask = dec_token_output['original_pad_masks']

    (enc_tokens, enc_mask) = check_seq_len(enc_tokens, enc_mask)
    (dec_tokens, dec_mask) = check_seq_len(dec_tokens, dec_mask)

    enc_token_ids = tokenizer.convert_tokens_to_ids(enc_tokens)
    dec_token_ids = tokenizer.convert_tokens_to_ids(dec_tokens)
    enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
    enc_pad_mask = torch.tensor(enc_mask,
                                dtype=torch.int64).transpose(0, 1)
    dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
    dec_pad_mask = torch.tensor(dec_mask,
                                dtype=torch.int64).transpose(0, 1)

    collate_output = {
        'encoder_input': enc_token_ids,
        'encoder_pad_mask': enc_pad_mask,
        'decoder_input': dec_token_ids[:-1, :],
        'decoder_pad_mask': dec_pad_mask[:-1, :],
        'target': dec_token_ids.clone()[1:, :],
        'target_pad_mask': dec_pad_mask.clone()[1:, :],
        'target_smiles': decoder_smiles,
        }

    return collate_output


class MoleculeDataset(Dataset):

    """Simple Molecule dataset that reads from a single DataFrame."""

    def __init__(self, df, split='train'):
        """
        Args:
            df (pandas.DataFrame): DataFrame object with RDKit molecules and lengths.
        """

        self.mols = df['canonical_smiles'].tolist()
        self.lengths = df['lengths'].tolist()
        self.aug = SMILESAugmenter()
        val_idxs = df.index[df['set'] == 'val'].tolist()
        test_idxs = df.index[df['set'] == 'test'].tolist()
        idxs = set(range(len(df.index)))
        train_idxs = idxs - set(val_idxs).union(set(test_idxs))
        idx_map = {'train': train_idxs, 'val': val_idxs,
                   'test': test_idxs}
        self.mols = [self.mols[idx] for idx in idx_map[split]]
        self.lengths = [self.lengths[idx] for idx in idx_map[split]]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        mol = self.mols[idx]
        try:
            enc_smi = self.aug(mol)
        except:
            enc_smi = mol
        try:
            dec_smi = self.aug(mol)
        except:
            dec_smi = mol
        output = {'encoder_smiles': enc_smi, 'decoder_smiles': dec_smi}
        return output


class MoleculeDataLoader(object):

    """Loads data from a csv file containing molecules."""

    def __init__(
        self,
        file_path,
        batch_size=32,
        num_buckets=20,
        num_workers=32,
        forward=True,
        ):

        self.df = pandas.read_csv(file_path)
        train_dataset = MoleculeDataset(self.df, split='train')
        val_dataset = MoleculeDataset(self.df, split='val')
        self.tokenizer = \
            MolEncTokeniser.from_vocab_file(DEFAULT_VOCAB_PATH, REGEX,
                DEFAULT_CHEM_TOKEN_START)

        world_size = \
            torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = \
            torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                True, rank, world_size)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=collate_fn)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                num_workers=num_workers, pin_memory=True,
                collate_fn=collate_fn)

    def get_data(self):
        return (self.train_loader, self.val_loader)

"""
The following fuctions & classes have been added for fine-tuning.
They could be tidied up but they do the job. -RB
"""

def collate_fn_input_only(batch):
    """ Adapted for classification tasks, where input=SMILES, output=Float. """

    encoder_smiles = [x['encoder_smiles'] for x in batch]
    enc_token_output = tokenizer.tokenise(encoder_smiles, mask=True,
            pad=True)
    target = torch.tensor([x['target'] for x in batch]).type(torch.float16)

    enc_mask = enc_token_output['masked_pad_masks']
    enc_tokens = enc_token_output['masked_tokens']

    (enc_tokens, enc_mask) = check_seq_len(enc_tokens, enc_mask)

    enc_token_ids = tokenizer.convert_tokens_to_ids(enc_tokens)
    enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1).type(torch.float16)
    enc_pad_mask = torch.tensor(enc_mask,
                                dtype=torch.int64).transpose(0, 1).type(torch.float16)

    collate_output = {
        'encoder_input': enc_token_ids,
        'encoder_pad_mask': enc_pad_mask,
        'target': target
        }

    return collate_output


class UsptoLoader(object):

    def __init__(
        self,
        uspto50_ds,
        batch_size=32,
        num_buckets=20,
        num_workers=32,
        ):

        train_dataset = Uspto_splitter(uspto50_ds, split='train')
        val_dataset = Uspto_splitter(uspto50_ds, split='val')
        self.tokenizer = \
            MolEncTokeniser.from_vocab_file(DEFAULT_VOCAB_PATH, REGEX,
                DEFAULT_CHEM_TOKEN_START)

        world_size = \
            torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = \
            torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                True, rank, world_size)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=collate_fn)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                num_workers=num_workers, pin_memory=True,
                collate_fn=collate_fn)

    def get_data(self):

        return (self.train_loader, self.val_loader)


class Uspto_splitter(Dataset):

    def __init__(self, uspto50_ds, split='train'):

        self.uspto50_ds = uspto50_ds

        if split == 'train':
            self.indices = uspto50_ds.train_idxs
        elif split == 'val':
            self.indices = uspto50_ds.val_idxs
        elif split == 'test':
            self.indices = uspto50_ds.test_idxs

        self.indices = list(self.indices)

    def __getitem__(self, idx):

        reactant_mol, product_mol = self.uspto50_ds.__getitem__(self.indices[idx])
        reactant_mol = Chem.MolToSmiles(reactant_mol)
        product_mol = Chem.MolToSmiles(product_mol)
        output = {'encoder_smiles': reactant_mol, 'decoder_smiles': product_mol}

        return output

    def __len__(self):

        return len(self.indices)


class MolOptLoader():

    def __init__(
        self,
        molopt_ds,
        batch_size=32,
        num_buckets=20,
        num_workers=32,
        ):
        
        train_dataset = MolOpt_splitter(molopt_ds, split='train')
        val_dataset = MolOpt_splitter(molopt_ds, split='val')

        world_size = \
            torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = \
            torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                True, rank, world_size)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=collate_fn)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                num_workers=num_workers, pin_memory=True,
                collate_fn=collate_fn)

    def get_data(self):

        return (self.train_loader, self.val_loader)
        

class MolOpt_splitter(Dataset):

    def __init__(self, molopt_ds, split='train'):

        self.molopt_ds = molopt_ds

        if split == 'train':
            self.indices = molopt_ds.train_idxs
        elif split == 'val':
            self.indices = molopt_ds.val_idxs
        elif split == 'test':
            self.indices = molopt_ds.test_idxs

        self.indices = list(self.indices)

    def __getitem__(self, idx):

        # Note - property values are not currently normalized, this may help training.
        reactant_mol, product_mol = self.molopt_ds.__getitem__(self.indices[idx])
        output = {'encoder_smiles': reactant_mol, 'decoder_smiles': product_mol}

        return output

    def __len__(self):

        return len(self.indices)


class MolPropLoader():

    def __init__(
        self,
        molprop_ds,
        batch_size=32,
        num_buckets=20,
        num_workers=32,
        ):
        
        train_dataset = MolProp_splitter(molprop_ds, split='train')
        val_dataset = MolProp_splitter(molprop_ds, split='val')

        world_size = \
            torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        rank = \
            torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        sampler = torch.utils.data.SequentialSampler(train_dataset)
        batch_sampler = DistributedBatchSampler(sampler, batch_size,
                True, rank, world_size)

        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=collate_fn_input_only)
        self.val_loader = torch.utils.data.DataLoader(val_dataset,
                num_workers=num_workers, pin_memory=True,
                collate_fn=collate_fn_input_only)

    def get_data(self):

        return (self.train_loader, self.val_loader)
        

class MolProp_splitter(Dataset):

    def __init__(self, molprop_ds, split='train'):

        self.molprop_ds = molprop_ds

        if split == 'train':
            self.indices = molprop_ds.train_idxs
        elif split == 'val':
            self.indices = molprop_ds.val_idxs
        elif split == 'test':
            self.indices = molprop_ds.test_idxs

        self.indices = list(self.indices)

    def __getitem__(self, idx):

        # Note - property values are not currently normalized, this may help training.
        reactant_mol, target = self.molprop_ds.__getitem__(self.indices[idx])
        target = float(target.split('|')[-1])
        reactant_mol = str.replace(reactant_mol, '|', '')
        output = {'encoder_smiles': reactant_mol, 'target': target}

        return output

    def __len__(self):

        return len(self.indices)