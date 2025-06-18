#!/bin/bash
# This script filters out audio files in a wav.scp that are shorter than a specified minimum duration or longer than a specified maximum duration.

if [ $# -ne 4 ]; then
    echo "Usage: $0 <source-wav-scp> <target-wav-scp> <min-duration> <max-duration>"
    echo "e.g.: $0 wav.scp wav.scp 0.1 10.0"
    echo "Use 'none' for min/max to ignore that bound."
    exit 1
fi

source_wav_scp=$1
target_wav_scp=$2
min_duration=$3
max_duration=$4

tmp_wav_scp="${target_wav_scp}.tmp"

while IFS= read -r line; do
    file="${line#* }"
    duration=$(soxi -D "$file")

    if [ "$min_duration" != "none" ] && [ "$(echo "$duration < $min_duration" | bc)" -eq 1 ]; then
        echo "$(date) Removing short audio file: $file (duration: $duration seconds) from $source_wav_scp"
        continue
    fi

    if [ "$max_duration" != "none" ] && [ "$(echo "$duration > $max_duration" | bc)" -eq 1 ]; then
        echo "$(date) Removing long audio file: $file (duration: $duration seconds) from $source_wav_scp"
        continue
    fi

    echo "$line" >>"$tmp_wav_scp"
done <"$source_wav_scp"

mv "$tmp_wav_scp" "$target_wav_scp"

echo "$(date) Filtered wav.scp: $target_wav_scp"
