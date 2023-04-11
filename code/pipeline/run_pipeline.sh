#!/bin/bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/lnet/aic/opt/cuda/cuda-11.8

source /home/pechmanp/master-thesis/code/czech-gec/venv-envs/.venv/bin/activate

python3 /home/pechmanp/master-thesis/code/czech-gec/pipeline/pipeline.py