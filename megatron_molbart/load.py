from molbart.tokeniser import MolEncTokeniser
from molbart.util import DEFAULT_CHEM_TOKEN_START
from molbart.util import REGEX
from molbart.util import DEFAULT_VOCAB_PATH
from megatron import print_rank_0, get_tensorboard_writer
from megatron.initialize import initialize_megatron
from megatron.model import get_params_for_weight_decay_optimization
from megatron.learning_rates import AnnealingLR
from megatron import mpu
from megatron import get_timers
from apex.optimizers import FusedAdam as Adam
from torch.optim import AdamW
from megatron_bart import MegatronBART
from molbart.decoder import DecodeSampler
import deepspeed
from megatron import get_args
import numpy as np
import torch
from molbart.models.pre_train import BARTModel
import os
import argparse

tokenizer = MolEncTokeniser.from_vocab_file(DEFAULT_VOCAB_PATH, REGEX,
        DEFAULT_CHEM_TOKEN_START)
num_batches_processed = 0
epochs = 0

def build_model_default(args):
    VOCAB_SIZE = len(tokenizer)
    MAX_SEQ_LEN = 512
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    sampler = DecodeSampler(tokenizer, MAX_SEQ_LEN)

    model = BARTModel(
        sampler,
        pad_token_idx,
        VOCAB_SIZE,
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.hidden_size * 4,
        0.1,
        0.1,
        'gelu',
        10000,
        MAX_SEQ_LEN,
        dropout=0.1,
        )
    return model


def build_model(args):

    VOCAB_SIZE = len(tokenizer)
    MAX_SEQ_LEN = 512
    pad_token_idx = tokenizer.vocab[tokenizer.pad_token]
    sampler = DecodeSampler(tokenizer, MAX_SEQ_LEN)

    model = MegatronBART(
        sampler,
        pad_token_idx,
        VOCAB_SIZE,
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.hidden_size * 4,
        MAX_SEQ_LEN,
        dropout=0.1,
        )

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters()
                   if p.requires_grad)

    print_rank_0('Number of parameters in MegatronBART: '
                 + str(count_parameters(model)))
    return model


def get_optimizer(model, args):
    param_groups = get_params_for_weight_decay_optimization(model)
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False
    optimizer = AdamW(param_groups, lr=args.lr,
                      weight_decay=args.weight_decay,
                      betas=(args.adam_beta1, args.adam_beta2))
    return optimizer


def get_learning_rate_scheduler(optimizer, args):

    # Add linear learning rate scheduler.

    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup * args.train_iters,
        total_iters=args.train_iters,
        decay_style=args.lr_decay_style,
        min_lr=args.min_lr,
        last_iter=0,
        use_checkpoint_lr_scheduler=False,
        override_lr_scheduler=False,
        )

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = build_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    print_rank_0('DeepSpeed is enabled.')

    # (mpu if args.pipe_parallel_size == 0 else None)
    localrankmpi = int(os.getenv('LOCAL_RANK', '0'))
    rankmpi = int(os.getenv('RANK', '0'))
    args.rank = rankmpi
    args.local_rank = localrankmpi
    (model, optimizer, _, lr_scheduler) = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=(mpu if args.pipe_parallel_size == 0 else None),
        dist_init_required=False,
        )

    return (model, optimizer, lr_scheduler)

def load_model():
    initialize_megatron()
    args = get_args()
    (model, optimizer, lr_scheduler) = setup_model_and_optimizer(args)
    ckpt = model.load_checkpoint(args.save)
    return ckpt

if __name__ == '__main__':
    ckpt = load_model()