FROM nvcr.io/nvidia/pytorch:20.11-py3
ARG GITHUB_ACCESS_TOKEN
ARG MEGAMOLBART_REPO="github.com/MolecularAI/MolBART.git"
ARG MEGAMOLBART_BRANCH="megatron-molbart-with-zinc"
ARG MEGAMOLBART_CHECKPOINT="megamolbart_checkpoints_20210427.tgz"

ENV UCX_LOG_LEVEL error
ENV PATH /opt/conda/bin:$PATH
ENV CUDA_HOME /usr/local/cuda
SHELL ["/bin/bash", "-c"]
RUN echo "source activate base" > /etc/bash.bashrc

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y wget git unzip tmux vim libxrender1 \
    && rm -rf /var/lib/apt/lists/*

## Environment setup
COPY conda/env.yml /tmp/.
RUN conda env update --name base -f /tmp/env.yml && conda clean -afy

## Apex
RUN cd /tmp && git clone https://github.com/NVIDIA/apex
RUN cd /tmp/apex/ \
    && /opt/conda/bin/python3 -m pip install -v \
        --disable-pip-version-check --no-cache-dir \
        --global-option="--cpp_ext" --global-option="--cuda_ext" ./

## PySMILES -- requirements handled by conda environment
RUN git clone https://github.com/MolecularAI/pysmilesutils.git --branch master /opt/pysmilesutils \
    && cd /opt/pysmilesutils; pip install .

## MegaMolBART and Megatron code
RUN git clone https://${GITHUB_ACCESS_TOKEN}@${MEGAMOLBART_REPO} --branch ${MEGAMOLBART_BRANCH} /opt/MolBART \
    && cd /opt/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism && pip install . \
    && cd /opt/MolBART; pip install .
ENV PYTHONPATH /opt/MolBART/megatron_molbart:/opt/MolBART

## MegaMolBART checkpoint and vocab -- TODO: update checkpoint
RUN mkdir -p /models/megamolbart \
    && wget --quiet -O /tmp/${MEGAMOLBART_CHECKPOINT} \
    http://rilango-work.nvidia.com/megatron_molbart/${MEGAMOLBART_CHECKPOINT} \
    && cp /opt/MolBART/bart_vocab.txt /models/megamolbart/bart_vocab.txt \
    && tar xzf /tmp/${MEGAMOLBART_CHECKPOINT} -C /models/megamolbart

