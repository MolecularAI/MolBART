# MolBART

The MolBART project aims to pre-train a BART transformer language model [[2]](#2) on molecular SMILES strings [[4]](#4) by optimising a de-noising objective[[2]](#2) as well as a chemical format transformation specific to the SMILES language (heteroencoding)[[3]](#3). Pre-training lead to improved generalisation, performance, training speed and validity on downstream fine-tuned tasks. The approach has been tested on downstream tasks such as reaction prediction, retrosynthetic prediction, molecular optimisation and molecular property prediction[[1]](#1).


## Installation

Firstly, Apex and pysmilesutils must be downloaded, then the project dependencies can be installed as follows:
- `conda create --name molbart rdkit -c rdkit`
- `conda install pytorch==1.8.0 torchvision cudatoolkit==11.1 -c pytorch`
- `conda install gcc_linux-64 gxx_linux-64 mpi4py`
- `pip install requirements.txt`
- `cd ../pysmilesutils && python setup.py install`


## Code

The codebase is broadly split into the following parts:
* Models
* Data helpers
* Tokenisation
* Decoding
* Scripts


### Models

The  `models.py` file contains a Pytorch Lightning implementation of the BART language model, as well as Pytorch Lightning implementations of models for downstream tasks.


### Data Helpers

The `dataset.py` file contains a number of classes used to load, batch and process the data before it is passed to the model. Classes which inherit from `_AbsDataset` are subclasses of Pytorch's `nn.utils.Dataset` and are simply used to store and split data (molecules, reactions, etc) into its relevant subset (train, val, test).

Our `_AbsDataModule` class inherits from Pytorch Lightning's `LightningDataModule` class, and its subclasses are used to augment, tokenise and tensorise the data before it passed to the model.

Finally, we include a `TokenSampler` class which categorises sequences into buckets based on their length, and is able to sample a different batch size of sequences from each bucket. This helps to ensure that the model sees approximately the same number of tokens on each batch, as well as dramatically improving training speed.


### Tokenisation

Our `tokenise.py` file includes the `MolEncTokeniser` class which is capable of random 'BERT-style' masking of tokens, as well as padding each batch of sequences to be the same length. The tokeniser makes use of the `SMILESTokenizer` from the `pysmilesutils` library for tokenising SMILES into their constituent atoms.


### Decoding

We include implementations of greedy and beam search decoding in the `decode.py` file. Both implementations make use of batch decoding for improved evaluation speeds. They do not, however, cache results from previous decodes, rather, they simply pass the entire sequence of tokens produced so far through the transformer decoder.


### Scripts

The repository includes the following scripts:
* `train.py` runs the pre-training 
* `fine_tune.py` runs fine-tuning on a specified task
* `evaluate.py` evaluates the performance of a fine-tuned model
* `build_tokeniser.py` creates a tokeniser from a dataset and stores it in a pickle file

Each script can be run using `python -m molbart.<scipt_name> <args>`.

See the ArgumentParser args in each file for more details on each argument.

To run on multiple GPUs use the `--gpus <num>` argument for the train or fine tune scripts. This will run the script with Pytorch Lightning's distributed data parallel (DDP) processing. Validation will be disabled when using DDP to ensure the GPUs stay synchronised and stop possible deadlocks from occurring.


## References

<a id="1">[1]</a>
Ross Irwin, Spyridon Dimitriadis, Jiazhen He, and Esben Jannik Bjerrum, 
"Chemformer: A Pre-Trained Transformer for Computational Chemistry", 
ChemRXiv (2021)

<a id="2">[2]</a>
Lewis, Mike, et al., 
"Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.", 
arXiv preprint arXiv:1910.13461 (2019).

<a id="3">[3]</a>
Esben J Bjerrum and Boris Sattarov, 
“Improving Chemical Autoencoder Latent Space and Molecular De Novo Generation Diversity with Heteroencoders”, 
Biomolecules, (2018) [http://doi.org/10.3390/biom8040131](http://doi.org/10.3390/biom8040131)

<a id="4">[4]</a>
Weininger, David., 
"SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules.", 
Journal of chemical information and computer sciences 28.1 (1988): 31-36.
