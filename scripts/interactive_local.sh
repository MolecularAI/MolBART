#!/bin/bash -l

##### Training / development on a local machine
### CONFIRMED WORKING ON A GV100 WITH 32GB RAM

### CONFIG ###
CONTAINER="nvcr.io/nvidian/clara-lifesciences/megamolbart:latest"
STORAGE_DIR="/home/mgill/storage/megatron"
CODE_DIR="/home/mgill/code/MolBART"
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_small # First ten data files
CONFIG_DIR=${CODE_DIR}/config
CHECKPOINT_DIR=${STORAGE_DIR}/checkpoints
DEEPSPEED_CONFIG_DIR=${CODE_DIR}/config
TENSORBOARD_DIR=${STORAGE_DIR}/tensorboard
MEGAMOLBART_CODE_DIR=${CODE_DIR}
export MEGATRON_CONFIG_PATH=${CONFIG_DIR}/config_megatron.sh

DATA_MOUNT=/data
CONFIG_MOUNT=/config
CHECKPOINT_MOUNT=/checkpoints
MEGATRON_CHECKPOINT_MOUNT=${CHECKPOINT_MOUNT}/megatron
DEEPSPEED_CONFIG_MOUNT=/deepspeed_config
TENSORBOARD_MOUNT=/tensorboard
WORKDIR=/opt/MolBART
CONFIG_DEEPSPEED_JSON_MOUNT=${DEEPSPEED_CONFIG_MOUNT}/config_deepspeed.json


### ZeRO CONFIG ###
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

### MEGATRON ###
export NOW=`date '+%F_%H:%M:%S'`
source $MEGATRON_CONFIG_PATH
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
--gas ${gas} \
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
--save ${MEGATRON_CHECKPOINT_MOUNT} \
--fp16 \
--tensorboard-dir ${TENSORBOARD_MOUNT}"

### DEEPSPEED ###

deepspeed_options=" \
--deepspeed \
--deepspeed_config ${CONFIG_DEEPSPEED_JSON_MOUNT} \
--zero-stage ${stage} \
--zero-reduce-bucket-size ${rbs} \
--zero-allgather-bucket-size ${agbs}"

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} --zero-reduce-scatter"
fi

### CHECKPOINTING ###

chkp_opt=" --checkpoint-activations --checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} --profile-backward"
fi

### RUN COMMAND ###
full_options="${megatron_options} ${deepspeed_options} ${chkp_opt}"
MOUNTS="--volume ${DATA_DIR}:${DATA_MOUNT}:ro --volume ${CONFIG_DIR}:${CONFIG_MOUNT}:ro --volume ${CHECKPOINT_DIR}:${CHECKPOINT_MOUNT} --volume ${DEEPSPEED_CONFIG_DIR}:${DEEPSPEED_CONFIG_MOUNT}:ro --volume ${TENSORBOARD_DIR}:${TENSORBOARD_MOUNT} --volume /etc/passwd:/etc/passwd:ro --volume /etc/group:/etc/group:ro "

# Mount development code
if [ -d ${MEGAMOLBART_CODE_DIR} ]; then
    echo "Mounting development code from ${MEGAMOLBART_CODE_DIR}"
    MOUNTS="${MOUNTS} --volume ${MEGAMOLBART_CODE_DIR}:${WORKDIR}"
fi

docker run -it --rm \
--network host \
--gpus all \
${MOUNTS} \
--workdir ${WORKDIR} \
--shm-size=1g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-u $(id -u ${USER}):$(id -g ${USER}) \
-e HOME=/workspace \
-e TF_CPP_MIN_LOG_LEVEL=3 \
-e full_options="${full_options}" \
${CONTAINER} bash
# Inside container run: python megatron_molbart/train.py --deepspeed --deepspeed_mpi ${full_options}
