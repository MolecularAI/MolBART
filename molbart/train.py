import math
import torch
import pickle
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from molbart.dataset import (
    Uspto50, 
    Chembl, 
    MoleculeDataset,
    ConcatMoleculeDataset,
    MoleculeDataModule
)
from molbart.models import BARTModel
from molbart.tokenise import MolEncTokeniser
from molbart.decode import DecodeSampler


DEFAULT_BATCH_SIZE = 32
DEFAULT_ACC_BATCHES = 1
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_MASK_PROB = 0.15
DEFAULT_D_MODEL = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FEEDFORWARD = 2048
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 10
DEFAULT_ACTIVATION = "gelu"
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 12
DEFAULT_AUGMENT = True

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


def build_dataset(args):
    chembl = Chembl(args.data_path)

    # Use a maximum of two molecules in sequence
    # dataset = ConcatMoleculeDataset(
    #     chembl,
    #     double_mol_prob=0.5,
    #     triple_mol_prob=0.0
    # )
    # return dataset

    return chembl


def build_datamodule(args, dataset, tokeniser):
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


def load_tokeniser(args):
    tokeniser_path = Path(args.tokeniser_path)
    file_handle = tokeniser_path.open("rb")
    tokeniser = pickle.load(file_handle)
    file_handle.close()

    tokeniser.mask_prob = args.mask_prob
    return tokeniser


def build_model(args, sampler, vocab_size, total_steps):
    if args.model_type == "bart":
        model = BARTModel(
            sampler,
            vocab_size=vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_feedforward=args.d_feedforward,
            lr=args.lr,
            weight_decay=args.weight_decay,
            activation=args.activation,
            num_steps=total_steps,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            mask_prob=args.mask_prob,
            augment=args.augment
        )
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    return model


def build_trainer(args):
    gpus = 1 if use_gpu else None
    epochs = args.epochs
    acc_batches = args.acc_batches

    # precision = 16 if use_gpu else 32
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
        callbacks=[lr_monitor, checkpoint_cb]
    )

    return trainer


def main(args):
    print("Building tokeniser...")
    tokeniser = load_tokeniser(args)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = build_dataset(args)
    print("Finished dataset.")

    print("Building data module...")
    dm = build_datamodule(args, dataset, tokeniser)
    print("Finished data module.")

    dm.setup()
    vocab_size = len(tokeniser)
    train_steps = math.ceil(len(dm.train_dataloader()) / args.acc_batches) * args.epochs
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)

    print("Building model...")
    model = build_model(args, sampler, vocab_size, train_steps)
    print("Finished model.")

    print("Building trainer...")
    trainer = build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to trainer")
    trainer.fit(model, dm)
    print("Finished training.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokeniser_path", type=str)
    parser.add_argument("--model_type", type=str)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--mask_prob", type=float, default=DEFAULT_MASK_PROB)
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num_heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--d_feedforward", type=float, default=DEFAULT_D_FEEDFORWARD)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)

    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=DEFAULT_AUGMENT)

    args = parser.parse_args()
    main(args)
