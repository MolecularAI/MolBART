#!/bin/bash
python -m molenc.train 
 --data_path ../data/chembl_27.pickle \
 --tokeniser_path ../tokenisers/mol_opt_tokeniser.pickle \
 --model_type bart \
 --lr 0.005 
