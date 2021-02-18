import torch
import pickle
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from molbart.models import BARTModel, ReactionPredModel
from molbart.dataset import Uspto50, UsptoMit, FineTuneReactionDataModule
from molbart.decode import DecodeSampler


DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_SEQ_LEN = 256

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
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs
    )
    return dm


def load_tokeniser(args):
    tokeniser_path = Path(args.tokeniser_path)
    file_handle = tokeniser_path.open("rb")
    tokeniser = pickle.load(file_handle)
    file_handle.close()
    return tokeniser


def load_model(args, sampler, pad_token_idx):
    pre_trained = BARTModel.load_from_checkpoint(
        args.pre_trained_path, 
        decode_sampler=sampler
    )
    model = ReactionPredModel.load_from_checkpoint(
        args.model_path,
        model=pre_trained,
        decode_sampler=sampler,
        pad_token_idx=pad_token_idx
    )
    return model


def build_trainer(args):
    gpus = 1 if use_gpu else None
    precision = 16 if use_gpu else 32
    logger = TensorBoardLogger("tb_logs", name="fine_tune_eval")
    trainer = Trainer( 
        gpus=gpus, 
        precision=precision,
        logger=logger
    )
    return trainer


def print_results(args, results):
    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results.items():
        print(f"{key:<25}{val:.4f}")


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

    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = load_model(args, sampler, pad_token_idx)
    print("Finished model.")

    print("Building trainer...")
    trainer = build_trainer(args)
    print("Finished trainer.")

    print("Evaluating model...")
    results = trainer.test(model, datamodule=dm)
    print_results(args, results[0])
    print("Finished evaluation.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokeniser_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--pre_trained_path", type=str)
    parser.add_argument("--dataset", type=str)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)

    args = parser.parse_args()
    main(args)
