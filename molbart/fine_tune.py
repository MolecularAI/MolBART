import os
import argparse

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.models.pre_train import BARTModel
from molbart.models.bart_fine_tune import ReactionBART


# Default training hyperparameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_ACC_BATCHES = 1
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 100
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "cycle"
DEFAULT_AUGMENT = None
DEFAULT_WARM_UP_STEPS = 8000
DEFAULT_TRAIN_TOKENS = None
DEFAULT_NUM_BUCKETS = 24
DEFAULT_LIMIT_VAL_BATCHES = 1.0


def load_model(args, sampler, vocab_size, total_steps, pad_token_idx):
    if args.model_type == "forward_prediction":
        forward_pred = True
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    # These args don't affect the model directly but will be saved by lightning as hparams
    # Tensorboard doesn't like None so we need to convert to string
    augment = "None" if args.augment is None else args.augment
    train_tokens = "None" if args.train_tokens is None else args.train_tokens
    num_buckets = "None" if args.num_buckets is None else args.num_buckets
    extra_args = {
        "batch_size": args.batch_size,
        "acc_batches": args.acc_batches,
        "epochs": args.epochs,
        "clip_grad": args.clip_grad,
        "augment": augment,
        "train_tokens": train_tokens,
        "num_buckets": num_buckets,
        "limit_val_batches": args.limit_val_batches
    }

    # If no model is given, use random init
    if args.model_path in ["none", "None"]:
        model = ReactionBART(
            sampler,
            pad_token_idx,
            vocab_size,
            util.DEFAULT_D_MODEL,
            util.DEFAULT_NUM_LAYERS,
            util.DEFAULT_NUM_HEADS,
            util.DEFAULT_D_FEEDFORWARD,
            args.lr,
            DEFAULT_WEIGHT_DECAY,
            util.DEFAULT_ACTIVATION,
            total_steps,
            util.DEFAULT_MAX_SEQ_LEN,
            args.schedule,
            util.DEFAULT_DROPOUT,
            args.warm_up_steps,
            **extra_args
        )
    else:
        model = ReactionBART.load_from_checkpoint(
            args.model_path,
            decode_sampler=sampler,
            pad_token_idx=pad_token_idx,
            vocab_size=vocab_size,
            num_steps=total_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            schedule=args.schedule,
            warm_up_steps=args.warm_up_steps,
            **extra_args
        )
    return model


def main(args):
    util.seed_everything(73)

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = util.build_dataset(args.dataset, args.data_path)
    print("Finished dataset.")

    print("Building data module...")
    dm = util.build_reaction_datamodule(args, dataset, tokeniser)
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // args.gpus
    dm._num_workers = num_workers
    print(f"Using {str(num_workers)} workers for data module.")
    print("Finished datamodule.")

    vocab_size = len(tokeniser)
    train_steps = util.calc_train_steps(args, dm)
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, util.DEFAULT_MAX_SEQ_LEN)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = load_model(args, sampler, vocab_size, train_steps + 1, pad_token_idx)
    sampler.max_seq_len = model.max_seq_len
    print("Finished model.")

    print("Building trainer...")
    trainer = util.build_trainer(args)
    print("Finished trainer.")

    print("Fitting data module to trainer")
    trainer.fit(model, dm)
    print("Finished training.")

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
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # Model and training args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--augment", type=str, default=DEFAULT_AUGMENT)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument("--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES)
    parser.add_argument("--gpus", type=int, default=util.DEFAULT_GPUS)

    args = parser.parse_args()
    main(args)
