# Megatron MolBART

This is a version of MolBART that uses NVIDIA's Megatron framework for model parallelism along with support for Multi-GPU and Multi-Node data parallelism using DeepSpeed.

## Installation

The project requires the `pysmilesutils` library to be installed (see README in pysmilesutils). MolBART also requires RDKit (although this should be installed as part of the installation procedure for pysmilesutils). The other requirements are listed in megatron_requirements.txt at the top level of this repository. Importantly, NVIDIA's Apex must be installed (follow the instructions on https://github.com/NVIDIA/apex). Also, DeepSpeed must be installed (```pip install deepspeed```). Last but not least, Megatron v1.1.5 must be installed as well. To do this, download the Megatron source compatible with DeepSpeed (https://github.com/microsoft/DeepSpeedExamples/tree/master/Megatron-LM-v1.1.5-3D_parallelism) and run ```python setup.py install```.


## Code

### megatron_bart.py

This file contains the source code for the Megatron implementation of the BART architecture. The high-level API of the MegatronBART model exactly resembles BARTModel. Concretely, the overall API is kept the same but the encoder and decoder is swapped out with parallelized versions.

### csv_data.py

This file contains a simple data loader for csv files that have SMILES strings on each row.

### train.py

This file contains the training code that sets up Megatron and DeepSpeed and runs the training loop. Currently, the code loads in a subset of ChEMBL in CSV format and runs training (line 320). Replace this with file with your own data.

### ds_config.json

This file contains the parameters for DeepSpeed (important). Notably, all of the parameters in this file should probably be kept the same except ```train_batch_size``` and ```train_micro_batch_size_per_gpu```. These parameters can be modified to improve training speed; generally, the larger batch sizes lead to much faster training. During training, DeepSpeed prints out the number of samples being processed per second (```SamplesPerSec```), which can be used a proxy for determining how fast training is.

### Kicking off Training

The `train_megatron.sh` bash script in the top level of the repository can be run to kick off training. Important parameters in this script include:

```GPUS_PER_NODE```: Number of GPUs to use per node <br>
```NNODES```: Number of nodes to run DeepSpeed with <br>
```MASTER_ADDR```: Distributed training address (must be changed for SLURM/Multi-Node setting) <br>
```MASTER_PORT```: Distributed training port (must be changed for SLURM/Multi-Node setting) <br>
```NODE_RANK```: Must be changed for SLURM/Multi-Node setting <br>
```mp_size```: Model parallelism size <br>
```--num-layers```: Number of hidden layers in Encoder and Decoder <br>
```--hidden-size```: Hidden dimension in Encoder and Decoder <br>
```--num-attention-heads```: Number of attention heads in Encoder and Decoder <br>
```train-iters```: How many training iterations <br>
```save```: Output model checkpoint directory name <br>

The default `train_megatron.sh` script in this repository runs the same architecture as the original MolBART model on 4 GPUs and a single node.

