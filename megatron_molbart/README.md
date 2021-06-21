# MegaMolBART

This is a version of MolBART that uses NVIDIA's Megatron framework for model parallelism along with support for Multi-GPU and Multi-Node data parallelism and model parallelism using DeepSpeed. 

## Installation

*NOTE:* these setup instructions are superceded by a Docker container and build scripts, which are currently being refined. We are also working to make the container available  

Follow these steps in order:

```
conda create -c rdkit -n molbart rdkit
conda activate molbart
pip install -r ../requirements.txt # MolBART repo
pip install -e .. # MolBART repo
pip install pybind11==2.6.2
pip install six==1.15.0
pip install regex
pip install deepspeed==0.3.10
cd apex; pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ # Download NVIDIA apex source
cd pysmilesutils; python setup.py install # Download pysmilesutils source
cd Megatron-LM-v1.1.5-3D_parallelism; python setup.py install
```

After all of these steps, if you still get an import error when running train_megatron.sh involving amp_C, this will fix the issue:

`pip uninstall apex; cd apex; pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`  

## Model Code

### megatron_bart.py

This file contains the source code for the Megatron implementation of the MolBART architecture. The high-level API of the MegatronBART model exactly resembles BARTModel. Concretely, the overall API is kept the same but the encoder and decoder is swapped out with parallelized versions.

### csv_data.py

This file contains a simple data loader for csv files that have SMILES strings on each row.

### train.py

This file contains the training code that sets up Megatron and DeepSpeed and runs the training loop. Currently, the code loads in a subset of the data in CSV format and runs training.

## Configuration and Training Files
### DeepSpeed Config File

The DeepSpeed config file `config_deepspeed.json` is located in the `config` directory in the base level of the repo.

These files contain the parameters for DeepSpeed. Notably, all of the parameters in this file should probably be kept the same except `train_micro_batch_size_per_gpu`, which is the batch size per model instance (see the [DeepSpeed config](https://www.deepspeed.ai/docs/config-json/) for more information). This parameter can be modified to balance training speed and model memory. Generally, larger batch sizes lead to much faster training, but require more memory. During training, DeepSpeed prints out the number of samples being processed per second (`SamplesPerSec`), which can be used a proxy for determining how fast training is.

### Megatron Config File

Several examples of the config file `config_megatron.sh` are located in the `config` directory in the base level of the repo.

### Training Scripts

Several versions of training scripts are located in the `scripts` directory in the base level of the repo:
* `run_training.sh` is for SLURM/Pyxis clusters with SBATCH. This has been tested in a variety of multi-node, model parallel, and data parallel settings.
* `interactive_slurm.sh` is for SLURM/Pyxis clusters with interactive development. This has only been tested on single node configurations with data parallel.
* `interactive_local.sh` is for interactive development without a scheduler (i.e. Docker only). This has only been tested on single GPU configurations.

The following parameters are important in the `run_training.sh` script:
* Number of nodes to use (`--nodes`)
* Number of GPUs to use per node (`--gpus-per-node`)
* Total number of tasks (`--ntasks`)
* Number of tasks per node (`--ntasks-per-node`)
* Distributed training address `MASTER_ADDR`
* Distributed training port: `MASTER_PORT`

### Model and Data Parallelism Gotchas

- As described in the sections above, there are two configuration files used for training -- one for Megatron (`config_megatron.sh`) and one for DeepSpeed (`config_deepspeed.json`). These are both located in the `config` folder in the base level of the repo. With the exception of `train_micro_batch_size_per_gpu`, the DeepSpeed config will remain largely unchanged.
- The parameter `mp_size` in the Megatron config file indicates the number of GPUs the model is split across for model parallelism.
- The parameter `WORLD_SIZE` is calculated in the training script (located in `scripts`). It must be equal to: No. of GPUS * No. Nodes / `mp_size`
- The number of GPUs for model parallelism must be an integer multiple of the number of GPUs used in total. For example, if 12 GPUs are used with `mp_size` = 4, there will be three copies of the model (data parallelism) and each model will occupy four GPUs (model parallelism)
- There appears to be some issue with resuming training from a checkpoint if the number of iterations will be increased from its original value. The error arises from learning rate scaling and needs further investigation. The relevant parameters are `train_iters` and `lr_decay_iters` in the Megatron config.

<img src="assets/mp.png" alt="model and data parallelism" width="700"/>

Diagram source: https://medium.com/@esaliya/model-parallelism-in-deep-learning-is-not-what-you-think-94d2f81e82ed

