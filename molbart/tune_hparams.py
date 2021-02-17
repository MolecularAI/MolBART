import math
import torch
import pickle
import optuna
import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from functools import partial

from molbart.models import BARTModel, ReactionPredModel
from molbart.dataset import Uspto50, FineTuneReactionDataModule
from molbart.decode import DecodeSampler


DEFAULT_TIMEOUT_HOURS = 28 * 24
DEFAULT_NUM_TRIALS = 100
DEFAULT_MAX_SEQ_LEN = 256
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EPOCHS = 100
DEFAULT_SWA_LR = None
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SCHEDULE = "const"
DEFAULT_AUGMENT = None

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()


def build_dataset(args):
    dataset = Uspto50(args.data_path)
    return dataset


def build_datamodule(args, dataset, tokeniser, batch_size):
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        batch_size,
        args.max_seq_len,
        forward_pred=True,
        augment=args.augment,
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


def load_model(args, sampler, vocab_size, total_steps, lr, batch_size):
    if args.model_type == "forward_prediction":
        forward_pred = True
    else:
        raise ValueError(f"Unknown model type {args.model_type}")

    pre_trained = BARTModel.load_from_checkpoint(args.model_path)
    model = ReactionPredModel(
        pre_trained,
        sampler,
        lr,
        args.weight_decay,
        total_steps,
        args.epochs,
        args.schedule,
        args.swa_lr,
        max_seq_len=args.max_seq_len,
        batch_size=batch_size,
        clip_grad=args.clip_grad,
        augment=args.augment
    )
    return model


def build_trainer(args, model, acc_batches):
    gpus = 1 if use_gpu else None
    precision = 16 if use_gpu else 32

    logger = TensorBoardLogger("tb_logs", name=args.study_name)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(monitor="val_molecular_accuracy", save_last=True)

    epochs = args.epochs
    trainer = Trainer(
        logger=logger, 
        gpus=gpus, 
        min_epochs=epochs, 
        max_epochs=epochs,
        precision=precision,
        accumulate_grad_batches=acc_batches,
        gradient_clip_val=args.clip_grad,
        callbacks=[lr_monitor, checkpoint_cb],
        progress_bar_refresh_rate=0,
        limit_val_batches=4
    )

    return trainer


def print_results(args, results):
    print(f"Results for model: {args.model_path}")
    print(f"{'Item':<25}Result")
    for key, val in results.items():
        print(f"{key:<25} {val:.4f}")


def objective(trial, study, study_name, args):
    print(f"Runnning trial {trial.number}")

    print("Building tokeniser...")
    tokeniser = load_tokeniser(args)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = build_dataset(args)
    print("Finished dataset.")

    lr = trial.suggest_loguniform("learning_rate", 0.00001, 0.01)
    batch_size = trial.suggest_int("batch_size", 4, 32, log=True)
    acc_batches = trial.suggest_int("acc_batches", 1, 16, log=True)

    print("Building data module...")
    dm = build_datamodule(args, dataset, tokeniser, batch_size)
    print("Finished datamodule.")

    dm.setup()
    vocab_size = len(tokeniser)
    train_steps = math.ceil(len(dm.train_dataloader()) / acc_batches) * args.epochs
    print(f"Train steps: {train_steps}")

    sampler = DecodeSampler(tokeniser, args.max_seq_len)

    print("Loading model...")
    model = load_model(args, sampler, vocab_size, train_steps + 1, lr, batch_size)
    print("Finished model.")

    print("Building trainer...")
    trainer = build_trainer(args, model, acc_batches)
    print("Finished trainer.")

    print("Training with hparams:")
    print(f"LR: {lr:.6f}")
    print(f"Batch size: {batch_size}")
    print(f"Acc batches: {acc_batches}")

    print("Fitting data module to trainer...")
    trainer.fit(model, dm)
    print("Finished training.")

    # Use greedy search to speed things up when testing
    model.test_sampling_alg = "greedy"

    print("Evaluating model...")
    results = trainer.test(model, datamodule=dm)
    print_results(args, results[0])
    print("Finished evaluation.")

    acc = results[0]["test_molecular_accuracy"]
    print(f"Achieved accuracy: {acc:.2f}")

    # Save study so far
    f = Path(f"{study_name}.study").open("wb")
    pickle.dump(study, f)
    f.close()

    return acc


def print_dict(coll):
    for key, val in coll.items():
        val = f"{val:.6f}" if type(val) == float else str(val)
        print(f"{key:<15} {val}")


def main(args):
    if args.study_name is None:
        args.study_name = f"{args.schedule}_{str(args.epochs)}"

    if args.study_path is None:
        study = optuna.create_study(direction="maximize", study_name=args.study_name)
    else:
        f = Path(args.study_path).open("rb")
        study = pickle.load(f)
        f.close()

    print(f"Running hparam tuning study {args.study_name}...")
    timeout_secs = args.timeout_hours * 60 * 60
    obj_fn = partial(objective, study=study, study_name=args.study_name, args=args)
    study.optimize(obj_fn, n_trials=args.num_trials, timeout=timeout_secs)
    print("Completed tuning.")

    print(f"Best params: {study.best_params}")
    print(f"Best accuracy: {study.best_value}")
    print(f"Best trial: {study.best_trial}")
    print_dict(study.best_params)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tokeniser_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--study_name", type=str)
    parser.add_argument("--study_path", default=None)
    parser.add_argument("--num_trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--timeout_hours", type=int, default=DEFAULT_TIMEOUT_HOURS)

    # Model args
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--swa_lr", type=float, default=DEFAULT_SWA_LR)
    parser.add_argument("--clip_grad", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--augment", type=str, default=DEFAULT_AUGMENT)

    args = parser.parse_args()
    main(args)
