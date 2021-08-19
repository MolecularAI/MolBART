FROM nvcr.io/nvidia/pytorch:20.11-py3
ARG MEGAMOLBART_BRANCH="megatron-molbart-with-zinc"

ENV UCX_LOG_LEVEL error
ENV PATH /opt/conda/bin:$PATH
ENV CUDA_HOME /usr/local/cuda
ENV TERM xterm
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
RUN git clone https://github.com/MolecularAI/MolBART.git --branch ${MEGAMOLBART_BRANCH} /opt/MolBART \
    && cd /opt/MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism && pip install . \
    && cd /opt/MolBART; pip install . \
    && mkdir -p /models/megamolbart \
    && cp /opt/MolBART/bart_vocab.txt /models/megamolbart/bart_vocab.txt
ENV PYTHONPATH /opt/MolBART/megatron_molbart:/opt/MolBART

