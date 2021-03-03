import math
import torch
import pickle
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from molbart.tokeniser import MolEncTokeniser
from molbart.models.bart_fine_tune import ReactionBART
from molbart.data.datasets import Chembl, Uspto50, UsptoMit, Zinc
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

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


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
        augment=args.augment,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.num_buckets,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs
    )
    return dm


def build_reaction_datamodule(args, dataset, tokeniser):
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        DEFAULT_MAX_SEQ_LEN,
        forward_pred=True,
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
    gpus = 1 if use_gpu else None
    epochs = args.epochs
    acc_batches = args.acc_batches
    precision = 32

    logger = TensorBoardLogger("tb_logs", name=args.model_type)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_molecular_accuracy", save_last=True)

    trainer = Trainer(
        logger=logger, 
        gpus=gpus, 
        min_epochs=epochs, 
        max_epochs=epochs,
        precision=precision,
        accumulate_grad_batches=acc_batches,
        gradient_clip_val=args.clip_grad,
        limit_val_batches=args.limit_val_batches,
        callbacks=[lr_monitor, checkpoint_cb]
    )
    return trainer


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
