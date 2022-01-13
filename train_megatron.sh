GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
config_json="$script_dir/config/config_deepspeed.json"

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
mp_size=1
# DeepSpeed Pipeline parallelism
pp_size=0

megatron_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --num-layers 2 \
        --hidden-size 64 \
        --num-attention-heads 8 \
        --seq-length 512 \
        --tensorboard-dir /home/hsirelkhatim/tensorboard \
        --max-position-embeddings 512 \
        --batch-size 32 \
        --gas 16 \
        --train-iters 320000 \
        --lr-decay-iters 320000 \
        --data-impl mmap \
        --distributed-backend nccl \
        --lr 0.0001 \
        --lr-decay-style cosine \
        --dataset_path /home/hsirelkhatim/zinc
        --min-lr 1.0e-5 \
        --weight-decay 0 \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 1000 \
        --eval-interval 100000 \
        --eval-iters 10 \
        --save megatron_molbart_100m_checkpoint
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

run_cmd="deepspeed --num_nodes ${DLWS_NUM_WORKER} --num_gpus ${DLWS_NUM_GPU_PER_WORKER} megatron_molbart/train.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
