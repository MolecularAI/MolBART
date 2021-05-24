#!/bin/bash -l

##### THIS SCRIPT RUNS SEE JOB 24087

### CONFIG ###
CONTAINER="nvcr.io#nvidian/clara-lifesciences/megamolbart:latest"

STORAGE_DIR="/gpfs/fs1/projects/ent_joc/users/mgill/megatron"
DATA_DIR=${STORAGE_DIR}/data/zinc_csv
CONFIG_DIR=${STORAGE_DIR}/config
CHECKPOINT_DIR=${STORAGE_DIR}/checkpoints
DEEPSPEED_CONFIG_DIR=${STORAGE_DIR}/config
TENSORBOARD_DIR=${STORAGE_DIR}/tensorboard
MEGAMOLBART_CODE_DIR=${STORAGE_DIR}/code/MolBART
export MEGATRON_CONFIG_PATH=${CONFIG_DIR}/config_megatron_checkpoint.sh

DATA_MOUNT=/data
CONFIG_MOUNT=/config
CHECKPOINT_MOUNT=/checkpoints
DEEPSPEED_CONFIG_MOUNT=/deepspeed_config
TENSORBOARD_MOUNT=/tensorboard
WORKDIR=/opt/MolBART
CONFIG_JSON_MOUNT=${DEEPSPEED_CONFIG_MOUNT}/config_deepspeed_checkpoint.json

full_options="${megatron_options} ${deepspeed_options} ${chkp_opt}"
MOUNTS="${MEGAMOLBART_CODE_DIR}:${WORKDIR},${DATA_DIR}:${DATA_MOUNT},${CONFIG_DIR}:${CONFIG_MOUNT},${CHECKPOINT_DIR}:${CHECKPOINT_MOUNT},${DEEPSPEED_CONFIG_DIR}:${DEEPSPEED_CONFIG_MOUNT},${TENSORBOARD_DIR}:${TENSORBOARD_MOUNT}"

srun \
--account ent_joc_model_mpnn_pyt \
--partition interactive \
--pty \
--nodes=1 \
--ntasks-per-node 1 \
--container-image ${CONTAINER} \
--container-workdir ${WORKDIR} \
--container-mounts ${MOUNTS} \
bash
