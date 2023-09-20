#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

input_file="$1"
output_file="$2"

if [ ! -f "$input_file" ]; then
    echo "Input file not found: $input_file"
    exit 1
fi

# Function to shuffle the words in a line
shuffle_line() {
    line="$1"
    # Split the line into an array of words
    words=($line)
    # Shuffle the array
    shuffled_words=($(shuf -e "${words[@]}"))
    # Join the shuffled words back into a line
    shuffled_line="${shuffled_words[*]}"
    echo "$shuffled_line"
}

# Process the input file and write shuffled content to the output file
while IFS= read -r line; do
    shuffled_line=$(shuffle_line "$line")
    echo "$shuffled_line" >> "$output_file"
done < "$input_file"

echo "Words shuffled and written to $output_file"

# Usage: bash shuffle_words.sh text.txt text_shuffled.txt