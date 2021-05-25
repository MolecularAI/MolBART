"""Test loading of a Megatron checkpoint"""
from pathlib import Path
import os
import torch

from megatron import get_args, print_rank_0
from megatron.initialize import initialize_megatron
import megatron.checkpointing as megatron_checkpointing
from checkpointing import load_checkpoint

from megatron_bart import MegatronBART
from utils import DEFAULT_CHEM_TOKEN_START
from utils import DEFAULT_VOCAB_PATH
from utils import DEFAULT_MAX_SEQ_LEN
from utils import REGEX

from molbart.decoder import DecodeSampler
from molbart.tokeniser import MolEncTokeniser

CHECKPOINTS_DIR = '/checkpoints/megatron'


def load_tokenizer(tokenizer_vocab_path):
    """Load tokenizer from vocab file
    Params:
        tokenizer_vocab_path: str, path to tokenizer vocab
    Returns:
        MolEncTokeniser tokenizer object
    """

    tokenizer_vocab_path = Path(tokenizer_vocab_path)
    tokenizer = MolEncTokeniser.from_vocab_file(
        tokenizer_vocab_path,
        REGEX,
        DEFAULT_CHEM_TOKEN_START)

    return tokenizer


def load_model(tokenizer, max_seq_len, args):
    """Load saved model checkpoint
    Params:
        tokenizer: MolEncTokeniser tokenizer object
        max_seq_len: int, maximum sequence length
        args: Megatron initialized arguments
    Returns:
        MegaMolBART trained model
    """

    vocab_size = len(tokenizer)
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    sampler = DecodeSampler(tokenizer, max_seq_len)
    model = MegatronBART(
                        sampler,
                        pad_token_idx,
                        vocab_size,
                        args.hidden_size,
                        args.num_layers,
                        args.num_attention_heads,
                        args.hidden_size * 4,
                        max_seq_len,
                        dropout=0.1,
                        )
    load_checkpoint(model, None, None)
    model = model.cuda()
    model.eval()
    return model


if __name__ == '__main__':

    iteration = 'latest'

    # TODO load args from megatron file
    args = {
        'num_layers': 4,
        'hidden_size': 256,
        'num_attention_heads': 8,
        'max_position_embeddings': DEFAULT_MAX_SEQ_LEN,
        'tokenizer_type': 'GPT2BPETokenizer',
        'vocab_file': DEFAULT_VOCAB_PATH,
        'load': CHECKPOINTS_DIR,
    }

    with torch.no_grad():
        initialize_megatron(args_defaults=args)

        # args = get_args()
        # Load args from checkpoint
        if iteration == 'latest':
            with open(os.path.join(CHECKPOINTS_DIR, 'latest_checkpointed_iteration.txt'), 'r') as fh:
                iteration = int(fh.read().strip())
        else:
            iteration = int(iteration)

        print_rank_0(f'Loading checkpoint from iteration {iteration}...')
        checkpoint_path = megatron_checkpointing.get_checkpoint_name(args['load'], iteration)
        ckpt = torch.load(checkpoint_path)
        args = ckpt['args']

        tokenizer = load_tokenizer(DEFAULT_VOCAB_PATH)
        model = load_model(tokenizer, DEFAULT_MAX_SEQ_LEN, args)
    