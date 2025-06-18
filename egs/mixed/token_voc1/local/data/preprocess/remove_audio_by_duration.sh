#!/bin/bash
# This script removes all audio files in a given directory that are shorter than a specified minimum duration or longer than a specified maximum duration, including its subdirectories.

if [ $# -ne 3 ]; then
    echo "Usage: $0 <data-dir> <min-duration> <max-duration>"
    echo "e.g.: $0 ../wav_dump 0.1 10.0"
    echo "Use 'none' for min/max to ignore that bound."
    exit 1
fi

dir=$1
min_duration=$2
max_duration=$3

find "$dir" -type f -name "*.wav" | while IFS= read -r file; do
    duration=$(soxi -D "$file")
    if [ "$min_duration" != "none" ] && awk "BEGIN {exit !($duration < $min_duration)}"; then
        # echo "Removing short audio file: $file (duration: $duration seconds)"
        rm "$file"
    elif [ "$max_duration" != "none" ] && awk "BEGIN {exit !($duration > $max_duration)}"; then
        # echo "Removing long audio file: $file (duration: $duration seconds)"
        rm "$file"
    fi
done
