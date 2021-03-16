import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.data.datamodules import FineTuneReactionDataModule


DEFAULT_BATCH_SIZE = 32
DEFAULT_TASK = "None"
DEFAULT_GPUS = 1
DEFAULT_AUGMENT = True
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = None


def build_trainer(args, limit_test_batches=1.0):
    logger = TensorBoardLogger("tb_logs", name=f"eval_{args.model_type}_{args.dataset}")
    trainer = Trainer(
        gpus=args.gpus,
        logger=logger,
        limit_test_batches=limit_test_batches
    )
    return trainer


def main(args):
    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = util.build_dataset(args.dataset, args.data_path)
    print("Finished dataset.")

    if args.model_type in ["forward_prediction", "bart"]:
        forward = True
    elif args.model_type == "backward_prediction":
        forward = False
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    print("Building data module...")
    if args.dataset in ["chembl", "zinc"]:
        dm = util.build_molecule_datamodule(args, dataset, tokeniser)
    elif args.dataset in ["uspto_mit", "pande"]:
        dm = util.build_reaction_datamodule(args, dataset, tokeniser, forward=forward)
    print("Finished datamodule.")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)

    print("Loading model...")
    model = util.load_bart(args, sampler)
    print("Finished model.")

    print("Building trainer...")
    limit_test_batches = 1.0
    if args.dataset == "zinc":
        limit_test_batches = 0.1

    trainer = build_trainer(args, limit_test_batches=limit_test_batches)
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
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_seq_len", type=int, default=util.DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPUS)

    parser.add_argument("--augment", dest="augment", action="store_true")
    parser.add_argument("--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=DEFAULT_AUGMENT)

    args = parser.parse_args()
    main(args)
