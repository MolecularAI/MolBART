#!/bin/bash
python -m molenc.fine_tune \
 --dataset uspto_mit \
 --data_path ../data/uspto_mixed.pickle \
 --tokeniser_path ../tokenisers/mol_opt_tokeniser.pickle \
 --model_path None \
 --model_type forward_prediction \
 --epochs 50 \
 --lr 2.0 \
 --schedule transformer \
 --batch_size 8 \
 --acc_batches 16

