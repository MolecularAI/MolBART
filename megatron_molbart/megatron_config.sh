#!/bin/bash
export n_gpus=4
export mp_size=1
export pp_size=0
export model_parallel_size=1
export pipe_parallel_size=0
export dataset_path="/storage/disk1/hassan/Zinc_csv_files"
export num_layers=4
export hidden_size=256
export num_attention_heads=8
export seq_length=512
export max_position_embeddings=512
export batch_size=320
export gas=16
export train_iters=320000
export lr_decay_iters=320000
export data_impl=mmap
export distributed_backend=nccl
export lr=0.0001
export lr_decay_style=cosine
export min_lr=1.0e-5
export weight_decay=0
export clip_grad=1.0
export warmup=0.01
export log_interval=1
export save_interval=1000
export eval_interval=100000
export eval_iters=10
export master_addr=localhost
export gradient_accumulation_steps=1
export checkpoint_directory="../checkpoints"
