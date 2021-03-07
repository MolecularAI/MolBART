import torch
import pickle
import argparse
from rdkit import Chem
from pathlib import Path

import molbart.util as util
from molbart.decoder import DecodeSampler
from molbart.models.pre_train import BARTModel
from molbart.models.bart_fine_tune import ReactionBART
from molbart.data.datasets import MoleculeDataset
from molbart.data.datamodules import MoleculeDataModule


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_BEAMS = 5


class SmilesError(Exception):
    def __init__(self, idx, smi):
        message = f"RDKit could not parse smiles {smi} at index {idx}"
        super().__init__(message)


def build_dataset(args):
    text = Path(args.reactants_path).read_text()
    smiles = text.split("\n")
    smiles = [smi for smi in smiles if smi != "" and smi is not None]
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles]

    # Check for parsing errors
    for idx, mol in enumerate(molecules):
        if mol is None:
            raise SmilesError(idx, smiles[idx])

    test_idxs = list(range(len(molecules)))
    dataset = MoleculeDataset(molecules, train_idxs=[], val_idxs=[], test_idxs=test_idxs)
    return dataset


def build_datamodule(args, dataset, tokeniser, max_seq_len):
    dm = MoleculeDataModule(
        dataset,
        tokeniser,
        args.batch_size,
        max_seq_len,
        val_idxs=dataset.val_idxs,
        test_idxs=dataset.test_idxs,
        augment=False
    )
    return dm


def predict(model, test_loader):
    device = "cuda:0" if util.use_gpu else "cpu"
    model = model.to(device)
    model.eval()

    smiles = []
    log_lhs = []

    for b_idx, batch in enumerate(test_loader):
        device_batch = {
            key: val.to(device) if type(val) == torch.Tensor else val for key, val in batch.items()
        }
        with torch.no_grad():
            smiles_batch, log_lhs_batch = model.sample_molecules(device_batch, sampling_alg="beam")

        smiles.extend(smiles_batch)
        log_lhs.extend(log_lhs_batch)

    return smiles, log_lhs


def write_predictions(args, smiles, log_lhs):
    output_str = ""
    for smiles_beams, log_lhs_beams in zip(smiles, log_lhs):
        for smi, log_lhs in zip(smiles_beams, log_lhs_beams):
            output_str += f"{smi},{str(log_lhs)}\n"

        output_str += "\n"

    p = Path(args.products_path)
    p.write_text(output_str)


def main(args):
    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(args.vocab_path, args.chem_token_start_idx)
    print("Finished tokeniser.")

    print("Reading dataset...")
    dataset = build_dataset(args)
    print("Finished dataset.")

    sampler = DecodeSampler(tokeniser, util.DEFAULT_MAX_SEQ_LEN)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    print("Loading model...")
    model = util.load_eval_model(args, sampler, pad_token_idx)
    model.num_beams = args.num_beams
    sampler.max_seq_len = model.max_seq_len
    print("Finished model.")

    print("Building data loader...")
    dm = build_datamodule(args, dataset, tokeniser, model.max_seq_len)
    dm.setup()
    test_loader = dm.test_dataloader()
    print("Finished loader.")

    print("Evaluating model...")
    smiles, log_lhs = predict(model, test_loader)
    write_predictions(args, smiles, log_lhs)
    print("Finished evaluation.")

    print("Printing unknown tokens...")
    tokeniser.print_unknown_tokens()
    print("Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Program level args
    parser.add_argument("--reactants_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--products_path", type=str)
    parser.add_argument("--vocab_path", type=str, default=util.DEFAULT_VOCAB_PATH)
    parser.add_argument("--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START)

    # Model args
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_beams", type=int, default=DEFAULT_NUM_BEAMS)

    args = parser.parse_args()
    main(args)
