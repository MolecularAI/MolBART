#!/bin/bash
python -m molbart.train \
 --dataset zinc \
 --data_path ../data/zinc \
 --model_type bart \
 --lr 1.0 \
 --schedule transformer \
 --d_model 512 \
 --num_layers 6 \
 --num_heads 8 \
 --d_feedforward 2048 \
 --gpus 2 \
 --epochs 1 \
 --task mask_aug \
 --batch_size 64
