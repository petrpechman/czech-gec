#!/bin/bash

# Define the range of max_length values
max_lengths=(32 64 96 128)

# Define the starting batch size and step size
start_batch_size=4
step_size=4
max_size=512

# Loop over max_length values
for max_length in "${max_lengths[@]}"; do
    # Loop over batch sizes
    for batch_size in $(seq $start_batch_size $step_size $max_size); do
        output=$(../../venv-envs/.venv/bin/python3.9 find_batch_sizes.py --batch-size=$batch_size --max-length=$max_length)
        
        # Check if the output is not empty
        if [ -n "$output" ]; then
            echo "$output"
        else
            echo "Finish fo max length: $max_length"
            continue
        fi
    done
done


# Usage: bash find_batches.sh