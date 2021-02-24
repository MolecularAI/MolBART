import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.data.datamodules import FineTuneReactionDataModule


DEFAULT_BATCH_SIZE = 32


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


def build_trainer(args):
    gpus = 1 if util.use_gpu else None
    precision = 32
    logger = TensorBoardLogger("tb_logs", name="fine_tune_eval")
    trainer = Trainer( 
        gpus=gpus, 
        precision=precision,
        logger=logger
    )
    return trainer


def main(args):
    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = util.build_dataset(args.dataset, args.data_path)
    print("Finished dataset.")

    print("Building data module...")
    dm = build_datamodule(args, dataset, tokeniser)
    print("Finished datamodule.")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = util.load_eval_model(args, sampler, pad_token_idx)
    print("Finished model.")

    print("Building trainer...")
    trainer = build_trainer(args)
    print("Finished trainer.")

    print("Evaluating model...")
    results = trainer.test(model, datamodule=dm)
    util.print_results(args, results[0])
    print("Finished evaluation.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)

    args = parser.parse_args()
    main(args)
