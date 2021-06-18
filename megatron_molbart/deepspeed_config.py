import argparse
import json
import sys
from megatron import print_rank_0

# TODO cleanup or remove

if '__name__' == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',  type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()

    print('...Reading ds_config file \n')
    with open('megatron_molbart/ds_config.json') as infile:
        data = json.load(infile)

    print('...Setting ds_config file \n')   
    data['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    data['train_micro_batch_size_per_gpu'] = args.batch_size
    data['train_batch_size'] = args.world_size*args.gradient_accumulation_steps*args.batch_size

    print('...Writing ds_config file \n')
    with open('megatron_molbart/ds_config.json', 'w') as outfile:
        json.dump(data, outfile)
