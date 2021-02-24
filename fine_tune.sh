#!/bin/bash
python -m molbart.fine_tune \
 --dataset uspto_mit \
 --data_path ../data/uspto_mixed.pickle \
 --model_path None \
 --model_type forward_prediction \
 --epochs 100 \
 --lr 2.0 \
 --schedule transformer \
 --batch_size 16 \
 --acc_batches 16 \
 --train_tokens 512 \
 --limit_val_batches 0.2
