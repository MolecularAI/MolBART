#!/bin/bash
python -m molbart.evaluate \
 --data_path ../data/uspto_mit.pickle \
 --model_path model.ckpt
 --dataset uspto_mit
