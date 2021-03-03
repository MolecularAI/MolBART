#!/bin/bash
python -m molbart.train \
 --dataset zinc \
 --data_path ../data/zinc \
 --model_type bart \
 --lr 0.003 \
 --epochs 1
