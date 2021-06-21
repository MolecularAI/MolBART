#!/bin/bash -l

##### Interactive training / development on a cluster with SLURM
# Tested only with single node, single GPU configuration

### CONFIG ###
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart:latest"
STORAGE_DIR="/gpfs/fs1/projects/ent_joc/users/mgill/megatron"
DATA_DIR=${STORAGE_DIR}/data/zinc_csv_small
CONFIG_DIR=${STORAGE_DIR}/config
CHECKPOINT_DIR=${STORAGE_DIR}/checkpoints
DEEPSPEED_CONFIG_DIR=${STORAGE_DIR}/config
TENSORBOARD_DIR=${STORAGE_DIR}/tensorboard
MEGAMOLBART_CODE_DIR=${STORAGE_DIR}/code/MolBART
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
--dataset_path ${DATA_MOUNT} \
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

MOUNTS="${DATA_DIR}:${DATA_MOUNT},${CONFIG_DIR}:${CONFIG_MOUNT},${CHECKPOINT_DIR}:${CHECKPOINT_MOUNT},${DEEPSPEED_CONFIG_DIR}:${DEEPSPEED_CONFIG_MOUNT},${TENSORBOARD_DIR}:${TENSORBOARD_MOUNT}"

# Mount development code
if [ -d ${MEGAMOLBART_CODE_DIR} ]; then
    echo "Mounting development code from ${MEGAMOLBART_CODE_DIR}"
    MOUNTS=${MOUNTS}",${MEGAMOLBART_CODE_DIR}:${WORKDIR}"
fi

srun \
--pty \
--account ent_joc_model_mpnn_pyt \
--partition interactive \
--export=full_options="${full_options}" \
--mpi=pmix \
--nodes 1 \
--ntasks 2 \
--ntasks-per-node 2 \
--gpus-per-node 2 \
--container-image ${CONTAINER} \
--container-mounts ${MOUNTS} \
--container-workdir ${WORKDIR} \
bash
# Inside container: python megatron_molbart/train.py --deepspeed --deepspeed_mpi ${full_options}
