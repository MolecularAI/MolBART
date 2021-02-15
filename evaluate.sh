#!/bin/bash
python -m molenc.evaluate \
 --data_path ../data/pande_dataset.pickle \
 --tokeniser_path ../tokenisers/chembl_concat.pickle \
 --model_path tb_logs/forward_prediction/version_3/checkpoints/epoch\=92.ckpt \
 --pre_trained_path tb_logs/transformer/version_3/checkpoints/epoch\=9.ckpt
