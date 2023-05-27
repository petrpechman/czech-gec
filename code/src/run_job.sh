#!/bin/bash

set -e

while getopts "c:g:m:p:" flag
    do
             case "${flag}" in
                    c) CPU=${OPTARG};;
                    g) GPU=${OPTARG};;
                    m) MEM=${OPTARG};;
                    p) PIPELINE=${OPTARG};;
             esac
    done

cd "./$PIPELINE"

sbatch -p gpu --cpus-per-gpu=$CPU --gpus=$GPU --mem=${MEM}G ../run_exp.sh $PIPELINE
