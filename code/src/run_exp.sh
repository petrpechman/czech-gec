#!/bin/bash

set -e

pipeline_dir="$1"

cd "./$pipeline_dir"

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/lnet/aic/opt/cuda/cuda-11.8

source "../venv-envs/.vens/bin/activate"

# source /home/pechmanp/DP/czech-gec/code/venv-envs/.venv/bin/activate

python3 "./$pipeline_dir/pipeline.py"