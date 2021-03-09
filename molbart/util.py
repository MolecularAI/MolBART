import os
import math
import torch
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from molbart.tokeniser import MolEncTokeniser
from molbart.models.bart_fine_tune import ReactionBART
from molbart.data.datasets import Chembl, Uspto50, UsptoMit, Zinc, ZincSlice
from molbart.data.datamodules import MoleculeDataModule, FineTuneReactionDataModule


# Default model hyperparams
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_ACTIVATION = "gelu"
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_DROPOUT = 0.1

DEFAULT_VOCAB_PATH = "bart_vocab.txt"
DEFAULT_CHEM_TOKEN_START = 272
REGEX = "\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

DEFAULT_GPUS = 1

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


class StepCheckpoint(Callback):
    def __init__(self, step_interval):
        super().__init__()

        if type(step_interval) != int:
            raise TypeError(f"step_interval must be of type int, got type {type(step_interval)}")

        self.step_interval = step_interval

    def on_batch_end(self, trainer, model):
        step = trainer.global_step
        if step % self.step_interval == 0:
            self._save_model(trainer, model, step)

    def _save_model(self, trainer, model, step):
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                save_dir = trainer.weights_save_path
            else:
                save_dir = trainer.logger.save_dir or trainer.default_root_dir

            version = (
                trainer.logger.version
                if isinstance(trainer.logger.version, str) else f"version_{trainer.logger.version}"
            )
            version, name = trainer.training_type_plugin.broadcast((version, trainer.logger.name))
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")

        else:
            ckpt_path = os.path.join(trainer.weights_save_path, "checkpoints")

        save_path = f"{ckpt_path}/step={str(step)}.ckpt"
        trainer.save_checkpoint(save_path)


def number_of_mols(data_path):
    path = Path(data_path)

    idx_file_mapping = []
    if path.is_dir():
        num_lines = 0
        for f in path.iterdir():
            text = f.read_text()
            num_mols = len(text.split("\n")) - 1
            idx_file_mapping.append((num_lines, num_lines + num_mols, f))
            num_lines += num_mols

    else:
        text = path.read_text()
        num_lines = len(text.split("\n"))
        idx_file_mapping.append((0, num_lines, path))

    return num_lines, idx_file_mapping


def read_df_slice(idxs, idx_file_mapping):
    """ Read a slice of the dataset from disk by looking up the required files in the mapping

    Args:
        idxs (List[int]): Contiguous list of indices into the full dataset of molecules to read 
        idx_file_mapping (dict): Mapping returned by number_of_mols function

    Returns:
        (pd.DataFrame): DataFrame of lines from dataset 
    """

    file_idx_map = {}

    curr_idx = 0
    for start, end, file_path in idx_file_mapping:
        while curr_idx < len(idxs) and start <= idxs[curr_idx] < end:
            file_idx_map.setdefault(str(file_path), [])
            file_idx_map[str(file_path)].append(idxs[curr_idx] - start)
            curr_idx += 1

    dfs = []
    for file_path, file_idxs in file_idx_map.items():
        file_df = pd.read_csv(Path(file_path))
        df = file_df.iloc[file_idxs]
        dfs.append(df)

    df_slice = pd.concat(dfs, ignore_index=True, copy=False)
    return df_slice


def read_zinc_slice(data_path, rank, num_gpus, batch_size):
    num_mols, idx_file_mapping = number_of_mols(data_path)
    rank_idxs = [idxs.tolist() for idxs in np.array_split(list(range(num_mols)), num_gpus)]

    # Drop last mols to ensure all processes have the same number of batches
    num_mols = min([len(idxs) for idxs in rank_idxs])
    num_mols = batch_size * (num_mols // batch_size)
    idxs = rank_idxs[rank][:num_mols]

    df_slice = read_df_slice(idxs, idx_file_mapping)
    print(f"Read {str(len(df_slice.index))} molecules for gpu {str(rank)}")
    dataset = ZincSlice(df_slice)
    return dataset


def build_dataset(dataset_type, data_path):
    if dataset_type == "pande":
        dataset = Uspto50(data_path)
        print("Using Pande group dataset.")
    elif dataset_type == "uspto_mit":
        dataset = UsptoMit(data_path)
        print("Using USPTO MIT dataset.")
    elif dataset_type == "chembl":
        dataset = Chembl(data_path)
        print("Using Chembl dataset.")
    elif dataset_type == "zinc":
        dataset = Zinc(data_path)
    else:
        raise ValueError(f"Unknown dataset {dataset_type}.")

    return dataset


def build_molecule_datamodule(args, dataset, tokeniser):
    dm = MoleculeDataModule(
        dataset, 
        tokeniser,
        args.batch_size,  
        args.max_seq_len,
        args.task,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.num_buckets,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        augment=args.augment
    )
    return dm


def build_reaction_datamodule(args, dataset, tokeniser, forward=True):
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        DEFAULT_MAX_SEQ_LEN,
        forward_pred=forward,
        augment=args.augment,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.num_buckets
    )
    return dm


def load_tokeniser(vocab_path, chem_token_start):
    tokeniser = MolEncTokeniser.from_vocab_file(vocab_path, REGEX, chem_token_start)
    return tokeniser


def build_trainer(args):
    logger = TensorBoardLogger("tb_logs", name=args.model_type)
    accelerator = "ddp" if args.gpus > 1 else None
    plugins = ["ddp_sharded"] if args.gpus > 1 else None
    replace_sampler_ddp = False if args.gpus > 1 else True

    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_molecular_accuracy", save_last=True)
    callbacks = [lr_monitor, checkpoint_cb]

    # Zinc is so big we need to checkpoint more frequently than every epoch
    # We should also run the validation set more frequently to observe the model's performance
    val_check_interval = 1.0
    if args.dataset == "zinc":
        checkpoint_freq = 100000
        intra_epoch_checkpoint = StepCheckpoint(checkpoint_freq)
        callbacks.append(intra_epoch_checkpoint)
        val_check_interval = checkpoint_freq

    print(f"Num gpus: {args.gpus}")
    print(f"Accelerator: {accelerator}")

    trainer = Trainer(
        accelerator=accelerator,
        logger=logger, 
        gpus=args.gpus, 
        min_epochs=args.epochs, 
        max_epochs=args.epochs,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.clip_grad,
        limit_val_batches=args.limit_val_batches,
        callbacks=callbacks,
        replace_sampler_ddp=replace_sampler_ddp,
        plugins=plugins,
        val_check_interval=val_check_interval
    )
    return trainer


def seed_everything(seed):
    pl.utilities.seed.seed_everything(seed)


def load_eval_model(args, sampler, pad_token_idx):
    model = ReactionBART.load_from_checkpoint(
        args.model_path,
        decode_sampler=sampler,
        pad_token_idx=pad_token_idx
    )
    model.eval()
    return model


def calc_train_steps(args, dm):
    dm.setup()
    train_steps = math.ceil(len(dm.train_dataloader()) / args.acc_batches) * args.epochs
    return train_steps


def print_results(args, results):
    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results.items():
        print(f"{key:<25} {val:.4f}")
