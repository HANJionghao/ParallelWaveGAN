#!/bin/bash

wav_scp=$1

if [ -z "$wav_scp" ]; then
    echo "Usage: $0 <wav_scp>"
    echo "Example: $0 /path/to/your/wav.scp"
    exit 1
fi

echo "Checking if all files in $wav_scp exist..."
file_missing=0
line_count=0
while read -r line; do
    line_count=$((line_count + 1))
    # utt_id wav_file
    wav_file="${line#* }"
    if [ ! -f "$wav_file" ]; then
        echo "Error: $line_count line: $wav_file does not exist"
        file_missing=1
    fi
done < "$wav_scp"
exit $file_missing
