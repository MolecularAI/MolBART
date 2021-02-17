import torch
import random

from molbart.tokenise import MolEncTokeniser


# Use dummy SMILES strings
smiles_data = [
    "CCO.Ccc",
    "CCClCCl",
    "C(=O)CBr"
]

example_tokens = [
    ["^", "C", "(", "=", "O", ")", "unknown", "&"], 
    ["^", "C", "C", "<SEP>", "C", "Br", "&"]
]

random.seed(a=1)


def test_create_vocab():
    tokeniser = MolEncTokeniser(smiles_data[2:3])
    expected = {
        " ": 0,
        "?": 1,
        "^": 2,
        "&": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "C": 6,
        "(": 7,
        "=": 8,
        "O": 9,
        ")": 10,
        "Br": 11
    }

    vocab = tokeniser.vocab

    assert expected == vocab


def test_pad_seqs_padding():
    seqs = [[1,2], [2,3,4,5], []]
    padded, _ = MolEncTokeniser._pad_seqs(seqs, " ")
    expected = [[1,2, " ", " "], [2,3,4,5], [" ", " ", " ", " "]]

    assert padded == expected


def test_pad_seqs_mask():
    seqs = [[1,2], [2,3,4,5], []]
    _, mask = MolEncTokeniser._pad_seqs(seqs, " ")
    expected_mask = [[0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]

    assert expected_mask == mask


def test_mask_tokens_empty_mask():
    tokeniser = MolEncTokeniser(smiles_data)
    masked, token_mask = tokeniser._mask_tokens(example_tokens, empty_mask=True)
    expected_sum = 0
    mask_sum = sum([sum(m) for m in token_mask])

    assert masked == example_tokens
    assert expected_sum == mask_sum


def test_mask_tokens_masking():
    tokeniser = MolEncTokeniser(smiles_data, mask_prob=0.4)
    masked, token_mask = tokeniser._mask_tokens(example_tokens)

    expected_masks = [
        [True, False, False, True, False, False, False, False],
        [False, False, False, True, False, False, True]
    ]

    assert expected_masks == token_mask


def test_convert_tokens_to_ids():
    tokeniser = MolEncTokeniser(smiles_data[2:3])
    ids = tokeniser.convert_tokens_to_ids(example_tokens)
    expected_ids = [[2, 6, 7, 8, 9, 10, 1, 3], [2, 6, 6, 5, 6, 11, 3]]

    assert expected_ids == ids


def test_tokenise_one_sentence():
    tokeniser = MolEncTokeniser(smiles_data)
    tokens = tokeniser.tokenise(smiles_data)
    expected = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]

    assert expected == tokens["original_tokens"]


def test_tokenise_two_sentences():
    tokeniser = MolEncTokeniser(smiles_data)
    tokens = tokeniser.tokenise(smiles_data, sents2=smiles_data)
    expected = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "<SEP>", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "<SEP>", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "<SEP>", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]
    expected_sent_masks = [
        ([0] * 9) + ([1] * 8),
        ([0] * 7) + ([1] * 6),
        ([0] * 9) + ([1] * 8),
    ]

    assert expected == tokens["original_tokens"]
    assert expected_sent_masks == tokens["sentence_masks"]


def test_tokenise_mask():
    tokeniser = MolEncTokeniser(smiles_data, mask_prob=0.4)
    tokens = tokeniser.tokenise(smiles_data, sents2=smiles_data, mask=True)
    expected_m_tokens = [
        ["^", "<MASK>", "<MASK>", "O", ".", "<MASK>", "<MASK>", "c", "<SEP>", "C", "C", "<MASK>", ")", "C", "c", "c", "&"],
        ["^", "<MASK>", "Br", "Cl", "C", "<MASK>", "<SEP>", "<MASK>", "C", "Cl", "<MASK>", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "<SEP>", "<MASK>", "(", "=", "O", ")", "<MASK>", "Br", "&"]
    ]
    expected_tokens = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "<SEP>", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "<SEP>", "C", "C", "Cl", "C", "Cl", "&"],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "<SEP>", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]

    assert expected_m_tokens == tokens["masked_tokens"]
    assert expected_tokens == tokens["original_tokens"]


def test_tokenise_padding():
    tokeniser = MolEncTokeniser(smiles_data)
    output = tokeniser.tokenise(smiles_data, sents2=smiles_data, pad=True)
    expected_tokens = [
        ["^", "C", "C", "O", ".", "C", "c", "c", "<SEP>", "C", "C", "O", ".", "C", "c", "c", "&"],
        ["^", "C", "C", "Cl", "C", "Cl", "<SEP>", "C", "C", "Cl", "C", "Cl", "&", " ", " ", " ", " "],
        ["^", "C", "(", "=", "O", ")", "C", "Br", "<SEP>", "C", "(", "=", "O", ")", "C", "Br", "&"]
    ]
    expected_pad_masks = [
        [0] * 17,
        ([0] * 13) + ([1] * 4),
        [0] * 17
    ]
    expected_sent_masks = [
        ([0] * 9) + ([1] * 8),
        ([0] * 7) + ([1] * 6) + ([0] * 4),
        ([0] * 9) + ([1] * 8),
    ]

    assert expected_tokens == output["original_tokens"]
    assert expected_pad_masks == output["pad_masks"]
    assert expected_sent_masks == output["sentence_masks"]
