# coding=utf-8

"""Input/output checkpointing."""

import os
import random
import sys
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP
from megatron import (get_args,
                      mpu,
                      print_rank_0)
import megatron.checkpointing as megatron_checkpointing

_CHECKPOINT_VERSION = None


def use_model_module(model):
    use_model = isinstance(model, torchDDP)
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except:
        pass
    else:
        use_model = use_model | isinstance(model, DeepSpeedEngine)
    return use_model


def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def save_megatron_checkpoint(iteration, model, optimizer, lr_scheduler):
    """Save a model checkpoint."""
    args = get_args()
    
    # Only rank zero of the data parallel writes to the disk.
    if use_model_module(model):
        model = model.module

    if mpu.get_data_parallel_rank() == 0:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 2.0
        state_dict['iteration'] = iteration
        state_dict['model'] = model.state_dict_for_save_checkpoint()

        # Optimizer stuff.
        if not args.no_save_optim:
            if optimizer is not None:
                state_dict['optimizer'] = optimizer.state_dict()
            if lr_scheduler is not None:
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict['random_rng_state'] = random.getstate()
            state_dict['np_rng_state'] = np.random.get_state()
            state_dict['torch_rng_state'] = torch.get_rng_state()
            state_dict['cuda_rng_state'] = torch.cuda.get_rng_state()
            state_dict['rng_tracker_states'] \
                = mpu.get_cuda_rng_tracker().get_states()

        # Save.
        checkpoint_name = megatron_checkpointing.get_checkpoint_name(args.save, iteration)
        print('global rank {} is saving checkpoint at iteration {:7d} to {}'.
            format(torch.distributed.get_rank(), iteration,
                    checkpoint_name))
        megatron_checkpointing.ensure_directory_exists(checkpoint_name)
        torch.save(state_dict, checkpoint_name)
        print('  successfully saved {}'.format(checkpoint_name))

    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = megatron_checkpointing.get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            f.write(str(iteration))
    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def load_deepspeed_iteration(checkpoint_path):
    """Load most recent iteration from DeepSpeed checkpoint"""
    iteration = 0

    iteration_path = os.path.join(checkpoint_path, 'latest')
    if os.path.exists(iteration_path):
        with open(iteration_path, 'r') as fh:
            iteration = fh.read().strip()
        iteration = int(iteration.replace('global_step', ''))
    
    return iteration


def load_checkpoint(model, optimizer, lr_scheduler, load_arg='load', iteration=0):
    """Load a model checkpoint and return the iteration."""
    args = get_args()
    load_dir = getattr(args, load_arg)

    if use_model_module(model):
        model = model.module
    # Read the tracker file and set the iteration.
    tracker_filename = megatron_checkpointing.get_checkpoint_tracker_filename(load_dir)

    # If no tracker file, return iretation zero.
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return 0

    # Otherwise, read the tracker file and either set the iteration or
    # mark it as a release checkpoint.
    if (not release) and (iteration == 0):
        with open(tracker_filename, 'r') as f:
            metastring = f.read().strip()
            try:
                iteration = int(metastring)
            except ValueError:
                release = metastring == 'release'
                if not release:
                    print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                        tracker_filename))
                    sys.exit()

    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    if args.deepspeed:
        checkpoint_name, state_dict = model.load_checkpoint(load_dir)

        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return iteration

    else:
        # Checkpoint.
        checkpoint_name = megatron_checkpointing.get_checkpoint_name(load_dir, iteration, release)
        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                torch.distributed.get_rank(), checkpoint_name))

        # Load the checkpoint.
        try:
            state_dict = torch.load(checkpoint_name, map_location='cpu')
        except ModuleNotFoundError:
            # For backward compatibility.
            print_rank_0(' > deserializing using the old code structure ...')
            sys.modules['fp16.loss_scaler'] = sys.modules[
                'megatron.fp16.loss_scaler']
            state_dict = torch.load(checkpoint_name, map_location='cpu')
            sys.modules.pop('fp16.loss_scaler', None)
        except BaseException:
            print_rank_0('could not load the checkpoint')
            sys.exit()
            # Model.

        model.load_state_dict(state_dict['model'])

        # Optimizer.
        if not release and not args.finetune and not args.no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(state_dict['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            except KeyError:
                print_rank_0(
                    'Unable to load optimizer from checkpoint {}. '
                    'Specify --no-load-optim or --finetune to prevent '
                    'attempting to load the optimizer state, '
                    'exiting ...'.format(checkpoint_name))
                sys.exit()

    # set checkpoint version
    megatron_checkpointing.set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    # Set iteration.
    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = state_dict['iteration']
        except KeyError:
            try:  # Backward compatible with older checkpoints
                iteration = state_dict['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but unable to load '
                             'iteration from checkpoint {}, exiting'.format(
                                 checkpoint_name))
                sys.exit()
 

    # Check arguments.
    if 'args' in state_dict:
        checkpoint_args = state_dict['args']
        # check_checkpoint_args(checkpoint_args) # Disabled this due to a few missing checkpoint arguments
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # rng states.
    if not release and not args.finetune and not args.no_load_rng:
        try:
            random.setstate(state_dict['random_rng_state'])
            np.random.set_state(state_dict['np_rng_state'])
            torch.set_rng_state(state_dict['torch_rng_state'])
            torch.cuda.set_rng_state(state_dict['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(
                state_dict['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the optimizer state, '
                         'exiting ...'.format(checkpoint_name))
            sys.exit()

    torch.distributed.barrier()
    if mpu.get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))

    return iteration


# TODO -- sample code to convert a DeepSpeed checkpoint to a Megatron one
# Code is untested with model parallel formats
def convert_deepspeeed_checkpoint_to_megatron(checkpoint_dir, iteration):
    """Convert a DeepSpeed checkpoint to a Megatron one"""
    args = get_args()

    # Setup paths
    input_path = os.path.join(checkpoint_dir, f'iter_{iteration:07}/mp_rank_00_model_states.pt')
    output_path = os.path.join(checkpoint_dr, f'iter_{iteration:07}/mp_rank_00/model_optim_rng.pt')
    latest_checkpointed_iteration = os.path.join(checkpoint_dir, 'latest_checkpointed_iteration.txt')

    # Check iteration
    state_dict = torch.load(input_path)
    iteration_state_dict = int(state_dict['iteration'])
    assert int(iteration) == iteration_state_dict

    # Create checkpoint iteration file
    with open(latest_checkpointed_iteration, 'w') as fh:
        fh.write(str(int(iteration)))

    # Fix state dict and write to correct path
    state_dict['model'] = state_dict['module']
    _ = state_dict.pop('module')
    # Leave args out for now --> "AttributeError: 'Namespace' object has no attribute 'padded_vocab_size'"
    # state_dict['args'] = args
    # state_dict['checkpoint_version'] = 2.0

    torch.save(state_dict, output_path)


# # Previous version for saving DeepSpeed checkpoints
# def save_ds_checkpoint(iteration, model, args):
#     """Save a model checkpoint."""

#     sd = {}
#     sd['iteration'] = iteration

#     # rng states.

#     if not args.no_save_rng:
#         sd['random_rng_state'] = random.getstate()
#         sd['np_rng_state'] = np.random.get_state()
#         sd['torch_rng_state'] = torch.get_rng_state()
#         sd['cuda_rng_state'] = torch.cuda.get_rng_state()
#         sd['rng_tracker_states'] = \
#             mpu.get_cuda_rng_tracker().get_states()

#     model.save_checkpoint(args.save, client_state=sd)
