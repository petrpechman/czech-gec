#!/bin/bash

set -e

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/lnet/aic/opt/cuda/cuda-11.8

source "/home/pechmanp/DP/czech-gec/code/venv-envs/.venv/bin/activate"

cd "/home/pechmanp/DP/czech-gec/code/src/transformer/"

python3 "/home/pechmanp/DP/czech-gec/code/src/transformer/evaluator.py"