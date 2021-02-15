import math
import torch
import pickle
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

from molenc.models import BARTModel, ReactionPredModel
from molenc.dataset import Uspto50, UsptoMit, FineTuneReactionDataModule
from molenc.decode import DecodeSampler


DEFAULT_BATCH_SIZE = 32
DEFAULT_ACC_BATCHES = 1
DEFAULT_MAX_SEQ_LEN = 384
DEFAULT_LR = 1e-5
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 100
DEFAULT_SWA_LR = None
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "cycle"
DEFAULT_AUGMENT = None
DEFAULT_WARM_UP_STEPS = 8000
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 24
DEFAULT_LIMIT_VAL_BATCHES = 1.0

D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FEEDFORWARD = 2048
ACTIVATION = "gelu"

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


def build_dataset(args):
    if args.dataset == "pande":
        dataset = Uspto50(args.data_path)
        print("Using Pande group dataset.")
    elif args.dataset == "uspto_mit":
        dataset = UsptoMit(args.data_path)
        print("Using USPTO MIT dataset.")
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")

    return dataset


def build_datamodule(args, dataset, tokeniser):
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        args.max_seq_len,
        forward_pred=True,
        augment=args.augment,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        train_token_batch_size=args.train_tokens,
        num_buckets=args.num_buckets
    )
    return dm


def load_tokeniser(args):
    tokeniser_path = Path(args.tokeniser_path)
    file_handle = tokeniser_path.open("rb")
    tokeniser = pickle.load(file_handle)
    file_handle.close()
    return tokeniser


def load_model(args, sampler, vocab_size, total_steps, pad_token_idx):
    if args.model_type == "forward_prediction":
        forward_pred = True
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    # If no model is given, use random init
    if args.model_path in ["none", "None"]:
        pre_trained = BARTModel(
            sampler,
            vocab_size=vocab_size,
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            d_feedforward=D_FEEDFORWARD,
            lr=args.lr,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            activation=ACTIVATION,
            num_steps=total_steps,
            max_seq_len=args.max_seq_len
        )
    else:
        pre_trained = BARTModel.load_from_checkpoint(args.model_path)

    model = ReactionPredModel(
        pre_trained,
        sampler,
        args.lr,
        args.weight_decay,
        total_steps,
        args.epochs,
        args.schedule,
        args.swa_lr,
        pad_token_idx,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        clip_grad=args.clip_grad,
        augment=args.augment,
        warm_up_steps=args.warm_up_steps,
        acc_batches=args.acc_batches
    )

    return model


def build_trainers(args, model):
    gpus = 1 if use_gpu else None
    acc_batches = args.acc_batches

    # precision = 16 if use_gpu else 32

    precision = 32

    logger = TensorBoardLogger("tb_logs", name=args.model_type)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_molecular_accuracy", save_last=True)

    epochs = args.epochs
    swa_trainer = None
    if args.swa_lr is not None:
        epochs = int(epochs * 0.75)
        swa_epochs = args.epochs - epochs
        swa_version = f"version_{logger.version}_swa"
        swa_logger = TensorBoardLogger("tb_logs", name=args.model_type, version=swa_version)
        swa_trainer = Trainer(
            logger=swa_logger,
            gpus=gpus, 
            min_epochs=swa_epochs, 
            max_epochs=swa_epochs,
            precision=precision,
            accumulate_grad_batches=acc_batches,
            gradient_clip_val=args.clip_grad,
            callbacks=[lr_monitor, checkpoint_cb]
        )

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

    return trainer, swa_trainer


def print_results(args, results):
    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results.items():
        print(f"{key:<25} {val:.4f}")


def main(args):
    print("Building tokeniser...")
    tokeniser = load_tokeniser(args)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = build_dataset(args)
    print("Finished dataset.")

    print("Building data module...")
    dm = build_datamodule(args, dataset, tokeniser)
    print("Finished datamodule.")

    dm.setup()
    vocab_size = len(tokeniser)
    train_steps = math.ceil(len(dm.train_dataloader()) / args.acc_batches) * args.epochs
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = load_model(args, sampler, vocab_size, train_steps + 1, pad_token_idx)
    print("Finished model.")

    print("Building trainer...")
    trainer, swa_trainer = build_trainers(args, model)
    print("Finished trainer.")

    print("Fitting data module to trainer")
    trainer.fit(model, dm)
    print("Finished training.")

    print("Evaluating model...")
    results = trainer.test(model, datamodule=dm)
    print_results(args, results[0])
    print("Finished evaluation.")

    if args.swa_lr is not None:
        print("Training SWA model...")
        model.set_swa()
        swa_trainer.fit(model, dm)
        print("Finished SWA model training.")

        print("Evaluating SWA model...")
        results = swa_trainer.test(model, datamodule=dm)
        print_results(args, results[0])
        print("Finished SWA evaluation.")

        # print("Evaluating non-SWA model for comparison...")
        # model.use_swa = False
        # results = swa_trainer.test(model, datamodule=dm)
        # print_results(args, results[0])
        # print("Finished non-SWA evaluation.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokeniser_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset", type=str)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--swa_lr", type=float, default=DEFAULT_SWA_LR)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--augment", type=str, default=DEFAULT_AUGMENT)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES)

    args = parser.parse_args()
    main(args)
