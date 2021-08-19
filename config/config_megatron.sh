# Data parallel version
export mp_size=1
export pp_size=0
export batch_size=512
export num_layers=4
export hidden_size=256
export num_attention_heads=8
export seq_length=512
export max_position_embeddings=512
export gas=16
export train_iters=610000
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
export save_interval=10000
export eval_interval=10000
export eval_iters=10