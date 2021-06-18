# Megatron MolBART

This is a version of MolBART that uses NVIDIA's Megatron framework for model parallelism along with support for Multi-GPU and Multi-Node data parallelism using DeepSpeed. 

## Installation

*NOTE:* these setup instructions are superceded by a Docker container and build scripts, which are currently being cleaned up. We are also working to make the container available  

Follow these steps in order:

`conda create -c rdkit -n molbart rdkit`  
`conda activate molbart`  
`pip install -r ../requirements.txt` (MolBART repo)  
`pip install -e ..` (MolBART repo)  
`pip install pybind11==2.6.2`  
`pip install six==1.15.0`  
`pip install regex`  
`pip install deepspeed==0.3.10`  
`cd apex; pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./` (Download NVIDIA apex source)  
`cd pysmilesutils; python setup.py install` (Download pysmilesutils source)  
`cd Megatron-LM-v1.1.5-3D_parallelism; python setup.py install`  

After all of these steps, if you still get an import error when running train_megatron.sh involving amp_C, this will fix the issue:

`pip uninstall apex; cd apex; pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`  

## Code

### megatron_bart.py

This file contains the source code for the Megatron implementation of the BART architecture. The high-level API of the MegatronBART model exactly resembles BARTModel. Concretely, the overall API is kept the same but the encoder and decoder is swapped out with parallelized versions.

### csv_data.py

This file contains a simple data loader for csv files that have SMILES strings on each row.

### train.py

This file contains the training code that sets up Megatron and DeepSpeed and runs the training loop. Currently, the code loads in a subset of ChEMBL in CSV format and runs training (line 320). Replace this file with your own data.

### DeepSpeed Config File

The config file `config_deepspeed.json` is located in the `config` directory in the base level of the repo.

These file contains the parameters for DeepSpeed (important). Notably, all of the parameters in this file should probably be kept the same except `train_batch_size` and `train_micro_batch_size_per_gpu`. These parameters can be modified to improve training speed; generally, the larger batch sizes lead to much faster training. During training, DeepSpeed prints out the number of samples being processed per second (`SamplesPerSec`), which can be used a proxy for determining how fast training is.

### Megatraon Config File

The config file `config_megatron.sh` is located in the `config` directory in the base level of the repo.

### Kicking off Training 

*NOTE:* this file is the original (legacy) version of the training file. Updated versions are located in the `scripts` directory in the base level of the repo.

The `train_megatron.sh` bash script in the top level of the repository can be run to kick off training. Important parameters in this script include:

`GPUS_PER_NODE`: Number of GPUs to use per node  
`NNODES`: Number of nodes to run DeepSpeed with  
`MASTER_ADDR`: Distributed training address (must be changed for SLURM/Multi-Node setting)  
`MASTER_PORT`: Distributed training port (must be changed for SLURM/Multi-Node setting)  
`NODE_RANK`: Must be changed for SLURM/Multi-Node setting  
`mp_size`: Model parallelism size  
`--num-layers`: Number of hidden layers in Encoder and Decoder  
`--hidden-size`: Hidden dimension in Encoder and Decoder  
`--num-attention-heads`: Number of attention heads in Encoder and Decoder  
`train-iters`: How many training iterations  
`save`: Output model checkpoint directory name  

The default `train_megatron.sh` script in this repository runs the original 12 million parameter MolBART model on 4 GPUs and a single node on the ChEMBL dataset.

### Model and Data Parallelism Gotchas

- The training scripts are located in the `scripts` folder in the base level of the repo
- There are two configuration files used for training -- one for Megatron (`config_megatron.sh`) and one for DeepSpeed (`config_deepspeed.json`). These are both located in the `config` folder in the base level of the repo
- The parameter `mp_size` in the Megatron config file indicates number of GPUs the model is split across for model parallelism
- The number of GPUs for model parallelism must be an integer multiple of the number of GPUs used for model parallelism
- Example: for 12 GPUs with `mp_size` = 4, there will be three copies of the model (data parallelism) and each model will occupy four GPUs (model parallelism)
- The parameter `WORLD_SIZE` is calculated in the SLURM training script. It must be equal to: No. of GPUS * No. Nodes / MP size
- In the DeepSpeed config file, the parameter `train_batch_size` = `WORLD_SIZE` * `train_micro_batch_size_per_gpu` * `gradient_accumulation_steps`
- If training will need to resume from a checkpoint, ensure that the number of iterations is set at least as large as what will be needed. Learning rate scaling depends on the maximum iteration number and is enabled by default. It will cause an error if the iteration number is increased beyond it's original value.


<img src="assets/mp.png" alt="model and data parallelism" width="700"/>

Diagram source: https://medium.com/@esaliya/model-parallelism-in-deep-learning-is-not-what-you-think-94d2f81e82ed

