#!/bin/bash
# This script removes all short .wav files from a data directory, including its subdirectories.

if [ $# -ne 2 ]; then
    echo "Usage: $0 <data-dir> <min-duration>"
    echo "e.g.: $0 ../wav_dump 0.1"
    exit 1
fi

dir=$1
min_duration=$2

find "$dir" -type f -name "*.wav" | while IFS= read -r file; do
    duration=$(soxi -D "$file") 

    if (( $(bc <<< "$duration < $min_duration") )); then
        # echo "Removing short audio file: $file (duration: $duration seconds)"
        rm "$file"
    fi
done
