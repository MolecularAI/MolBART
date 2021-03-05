from molbart.tokeniser import MolEncTokeniser
from molbart.util import DEFAULT_CHEM_TOKEN_START
from molbart.util import REGEX
from molbart.util import DEFAULT_VOCAB_PATH
from megatron import print_rank_0
from megatron.initialize import initialize_megatron
from megatron.model import get_params_for_weight_decay_optimization
from megatron.learning_rates import AnnealingLR
from megatron import mpu
from megatron.utils import report_memory
from megatron.utils import reduce_losses
from megatron import get_timers
from apex.optimizers import FusedAdam as Adam
from torch.optim import AdamW
from megatron_bart import MegatronBART
from molbart.decoder import DecodeSampler
import deepspeed
from csv_data import MoleculeDataLoader
from megatron import get_args
import numpy as np
import pickle
import torch
from molbart.models.pre_train import BARTModel
import random
from deepspeed.utils import RepeatingLoader
import os

tokenizer = MolEncTokeniser.from_vocab_file(DEFAULT_VOCAB_PATH, REGEX,
        DEFAULT_CHEM_TOKEN_START)


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
	    "gelu",
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
    #(mpu if args.pipe_parallel_size == 0 else None)
    (model, optimizer, _, lr_scheduler) = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        mpu=(mpu if args.pipe_parallel_size == 0 else None),
        dist_init_required=False,
        )

    return (model, optimizer, lr_scheduler)


def get_batch(data_iterator):
    """Generate a batch"""

    keys = [
        'encoder_input',
        'encoder_pad_mask',
        'decoder_input',
        'decoder_pad_mask',
        'target',
        'target_pad_mask',
        ]
    datatype = torch.int64
    data = next(data_iterator)
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    encoder_tokens = data_b['encoder_input'].long()
    encoder_pad_mask = data_b['encoder_pad_mask'].bool()
    decoder_tokens = data_b['decoder_input'].long()
    decoder_pad_mask = data_b['decoder_pad_mask'].bool()
    target = data_b['target'].long()
    target_pad_mask = data_b['target_pad_mask'].long()

    return {
        'encoder_input': encoder_tokens,
        'encoder_pad_mask': encoder_pad_mask,
        'decoder_input': decoder_tokens,
        'decoder_pad_mask': decoder_pad_mask,
        'target': target,
        'target_pad_mask': target_pad_mask,
        }


def forward_step(data_iterator, model):
    """Forward step."""

    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    batch = get_batch(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    tokens = batch['target']
    pad_mask = batch['target_pad_mask']
    outputs = model(batch)
    token_output = outputs['token_output']
    loss = model.module._calc_loss(batch, outputs)
    acc = model.module._calc_char_acc(batch, outputs)

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])
    return (loss, {'mask loss': reduced_loss[0], 'acc': acc})


def backward_step(optimizer, model, loss):
    """Backward step."""

    timers = get_timers()

    # Backward pass.
    timers('backward-backward').start()
    model.backward(loss)
    timers('backward-backward').stop()
    timers('backward-allreduce').reset()


def eval_step(data_iterator, model):
    """Forward step."""

    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    batch = next(data_iterator)
    timers('batch generator').stop()

    # Forward model.
    val_ouputs = model.module.validation_step(batch)
    invalid_smiles = val_ouputs['val_invalid_smiles']

    # Reduce loss for logging.
    reduced_invalid_smiles = reduce_losses([invalid_smiles])
    return {'val_invalid_smiles': reduced_invalid_smiles[0]}


def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,
    pipe_parallel_size,
    ):
    """Single training step."""

    timers = get_timers()

    # Forward model for one step.
    timers('forward').start()
    (loss, loss_reduced) = forward_step_func(data_iterator, model)
    timers('forward').stop()

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    backward_step(optimizer, model, loss)
    timers('backward').stop()

    # Update parameters.
    timers('optimizer').start()
    model.step()
    timers('optimizer').stop()

    return loss_reduced


def save_ds_checkpoint(iteration, model, args):
    """Save a model checkpoint."""

    sd = {}
    sd['iteration'] = iteration
    # rng states.
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()
        sd['torch_rng_state'] = torch.get_rng_state()
        sd['cuda_rng_state'] = torch.cuda.get_rng_state()
        sd['rng_tracker_states'] = mpu.get_cuda_rng_tracker().get_states()

    model.save_checkpoint(args.save, client_state=sd)


def train(
    forward_step_func,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    val_data_iterator,
    pipe_parallel_size,
    args,
    ):
    """Train the model function."""

    timers = get_timers()
    model.train()
    iteration = 0
    timers('interval time').start()
    report_memory_flag = True
    while iteration < args.train_iters:
        loss = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            lr_scheduler,
            pipe_parallel_size,
            )
        
        iteration += 1
        print_rank_0('Iteration: ' + str(iteration) + '/'
                     + str(args.train_iters) + ', Loss: '
                     + str(loss['mask loss'].item()) + ', Acc: '
                     + str(loss['acc']))

        if iteration % args.eval_interval == 0:
            val_loss = eval_step(train_data_iterator, model)
            print_rank_0('Iteration: ' + str(iteration) + '/'
                         + str(args.train_iters)
                         + ', Validation Invalid SMILES: '
                         + str(val_loss['val_invalid_smiles']))

        # Checkpointing
        if iteration % args.save_interval == 0:
        	save_ds_checkpoint(iteration, model, args)

    return iteration


def run_training():
    initialize_megatron()
    args = get_args()
    print_rank_0('Loading ChEMBL dataset ...')
    path = os.path.dirname(os.path.realpath(__file__))
    loader = MoleculeDataLoader(path+'/test_data/chembl_subset.csv', batch_size=32, num_workers=20)
    train_dataloader, val_dataloader = loader.get_data()
    print_rank_0('Setting up model ...')
    (model, optimizer, lr_scheduler) = \
        setup_model_and_optimizer(args)
    print_rank_0('Starting training ...')
    train_dataloader = RepeatingLoader(train_dataloader)
    val_dataloader = RepeatingLoader(val_dataloader)
    train(
        forward_step,
        model,
        optimizer,
        lr_scheduler,
        iter(train_dataloader),
        iter(val_dataloader),
        args.pipe_parallel_size,
        args,
        )

def load_model():
	initialize_megatron()
	args = get_args()
	(model, optimizer, lr_scheduler) = \
	    setup_model_and_optimizer(args)
	ckpt = model.load_checkpoint(args.save)

if __name__ == '__main__':
    run_training()

