#!/bin/bash

# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
export NNODES=1
export NODE_RANK=0

#export dataset_path=$1
export megatron_config_path=$1
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json=$3
#megatron_config=$1
if [ -z "$config_json" ]
then
      config_json="$script_dir/megatron_molbart/ds_config.json"
else
      megatron_config=$config_json
fi

if [ -z "$megatron_config_path" ]
then
      megatron_config="$script_dir/megatron_molbart/megatron_config.sh"
else
      megatron_config=$megatron_config_path
fi
#"
export GPUS_PER_NODE=4
source $megatron_config
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

run_python_config="python deepspeed_config.py --batch_size ${batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} --world_size ${WORLD_SIZE}"
eval ${run_python_config}

#ZeRO Configs
stage=1
reduce_scatter=true
contigious_gradients=false
rbs=50000000
agbs=5000000000

chkp_layers=1
PA=true
PA_CPU=false
CC=true
SYNCHRONIZE=true
PROFILE=false

# Megatron Model Parallelism
#mp_size=1
# DeepSpeed Pipeline parallelism
#pp_size=0

megatron_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --dataset_path ${dataset_path} \
        --num-layers ${num_layers} \
        --hidden-size ${hidden_size} \
        --num-attention-heads ${num_attention_heads} \
        --seq-length ${seq_length} \
        --max-position-embeddings ${max_position_embeddings} \
        --batch-size ${batch_size} \
        --gas ${gas}\
        --train-iters ${train_iters} \
        --lr-decay-iters ${lr_decay_iters} \
        --data-impl ${data_impl} \
        --distributed-backend ${distributed_backend} \
        --lr ${lr} \
        --lr-decay-style ${lr_decay_style} \

        --min-lr ${min_lr} \
        --weight-decay ${weight_decay} \
        --clip-grad ${clip_grad} \
        --warmup ${warmup} \
        --checkpoint-activations \
        --log-interval ${log_interval} \
        --save-interval ${save_interval} \
        --eval-interval ${eval_interval} \
        --eval-iters ${eval_iters} \
        --save ${checkpoint_directory}
        --fp16
"

deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${megatron_options} ${deepspeed_options} ${chkp_opt}"
#run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} megatron_molbart/train.py $@ ${full_options}"
run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} megatron_molbart/train.py --deepspeed --deepspeed_mpi ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
