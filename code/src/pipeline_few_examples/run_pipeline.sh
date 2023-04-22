#!/bin/bash
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/lnet/aic/opt/cuda/cuda-11.8

source /home/pechmanp/DP/czech-gec/code/venv-envs/.venv/bin/activate

python3 /home/pechmanp/DP/czech-gec/code/pipeline_few_examples/pipeline.py 