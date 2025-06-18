#!/bin/bash
# This script checks if the utterances in two files are aligned.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file1> <file2>"
    exit 1
fi

file1="$1"
file2="$2"
if [ ! -f "$file1" ]; then
    echo "File $file1 does not exist."
    exit 1
fi
if [ ! -f "$file2" ]; then
    echo "File $file2 does not exist."
    exit 1
fi

lines1=$(wc -l <"$file1")
lines2=$(wc -l <"$file2")
if [ "$lines1" -ne "$lines2" ]; then
    echo "Files have different number of lines between ${file1} and ${file2}: ${lines1} vs ${lines2}"
    exit 1
fi

lineno=0
while IFS= read -r line1 && IFS= read -r line2 <&3; do
    lineno=$((lineno + 1))
    utt_id1="${line1%% *}"
    utt_id2="${line2%% *}"
    if [ "$utt_id1" != "$utt_id2" ]; then
        echo "Utterance IDs do not match in ${file1} and ${file2} at line ${lineno}: ${utt_id1} vs ${utt_id2}"
        exit 1
    fi
done <"$file1" 3<"$file2"
