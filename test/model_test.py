import pytest
import torch
import random

from molbart.models import BARTModel
from molbart.tokenise import MolEncTokeniser
from molbart.decode import DecodeSampler


# Use dummy SMILES strings
react_data = [
    "CCO.C",
    "CCCl",
    "C(=O)CBr"
]

# Use dummy SMILES strings
prod_data = [
    "cc",
    "CCl",
    "CBr"
]

model_args = {
    "max_seq_len": 40,
    "d_model": 5,
    "num_layers": 2,
    "num_heads": 1,
    "d_feedforward": 32,
    "lr": 0.0001,
    "weight_decay": 0.0,
    "activation": "gelu",
    "num_steps": 1000
}

random.seed(a=1)
torch.manual_seed(1)


def build_tokeniser():
    tokeniser = MolEncTokeniser(react_data + prod_data)
    return tokeniser


def test_pos_emb_shape():
    tokeniser = build_tokeniser()
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, len(tokeniser), **model_args)
    pos_embs = model._positional_embs()

    assert pos_embs.shape[0] == model_args["max_seq_len"]
    assert pos_embs.shape[1] == model.d_model


def test_construct_input_shape():
    tokeniser = build_tokeniser()
    sampler = DecodeSampler(tokeniser, model_args["max_seq_len"])
    model = BARTModel(sampler, len(tokeniser), **model_args)

    token_output = tokeniser.tokenise(react_data, sents2=prod_data, pad=True)
    tokens = token_output["original_tokens"]
    sent_masks = token_output["sentence_masks"]

    token_ids = torch.tensor(tokeniser.convert_tokens_to_ids(tokens)).transpose(0, 1)
    sent_masks = torch.tensor(sent_masks).transpose(0, 1)

    emb = model._construct_input(token_ids, sent_masks)

    assert emb.shape[0] == max([len(ts) for ts in tokens])
    assert emb.shape[1] == 3
    assert emb.shape[2] == model_args["d_model"]
