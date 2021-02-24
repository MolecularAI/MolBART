#!/bin/bash
python -m molbart.train 
 --data_path ../data/chembl_27.pickle \
 --model_type bart \
 --lr 0.003 \
 --epochs 20 \
 --train_tokens 4096
