#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <filename> <interval>"
    exit 1
fi

filename="$1"
interval="$2"

if [ ! -f "$filename" ]; then
    echo "File not found: $filename"
    exit 1
fi

if ! [ "$interval" -eq "$interval" ] 2>/dev/null; then
    echo "Invalid interval: $interval"
    exit 1
fi

temp_file=$(mktemp)

while IFS= read -r line; do
    length=${#line}
    interval_start=$(( (length / interval) * interval ))
    interval_end=$(( interval_start + interval ))
    echo "$interval_start-$interval_end"
done < "$filename" | sort | uniq -c > "$temp_file"


total_count=$(wc -l $filename | awk '{ print $1 }')
# echo "Line Length Interval : Number of Lines"
while read -r count interval; do
    percent=$(echo "scale=8; ($count / $total_count) * 100" | bc)
    echo "$interval : $count : $percent"
done < "$temp_file"

rm "$temp_file"

# Usage:  bash compute_lengths.sh sentence.input 32 | sort -n