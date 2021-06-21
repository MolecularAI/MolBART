# Model parallel version
export mp_size=16
export pp_size=0
export batch_size=1
export num_layers=200
export hidden_size=512
export num_attention_heads=32
export seq_length=512
export max_position_embeddings=512
export gas=16
export train_iters=110000
export lr_decay_iters=110000
export data_impl=mmap
export distributed_backend=nccl
export lr=0.0001
export lr_decay_style=cosine
export min_lr=1.0e-5
export weight_decay=0
export clip_grad=1.0
export warmup=0.01
export log_interval=1
export save_interval=50
export eval_interval=50
export eval_iters=10
# export master_addr=$(bash -c 'hostname' | sort | head -1 | awk '{print $2}')
# export NODE_RANK=$SLURM_NODEID
# export NNODES=$SLURM_JOB_NUM_NODES
# export n_gpus=2
# export checkpoint_directory="/checkpoints/megatron"
# export dataset_path="/data"
# export gradient_accumulation_steps=1
# export model_parallel_size=1
# export pipe_parallel_size=0
