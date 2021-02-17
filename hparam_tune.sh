#!/bin/bash
python -m molbart.tune_hparams \
 --data_path ../data/pande_dataset.pickle \
 --tokeniser_path ../tokenisers/chembl_concat.pickle \
 --model_path saved_models/transformer/version_3/checkpoints/epoch\=9.ckpt \
 --model_type forward_prediction \
 --epochs 200 \
 --schedule const \
 --study_path const_200.study \
 --timeout_hours 300

