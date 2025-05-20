#!/bin/bash

scp=$1
echo "Checking for duplicate lines in ${scp}..."
duplicates=$(awk '{count[$0]++} END {for (line in count) if (count[line] > 1) print count[line], line}' "$scp")
if [ -z "$duplicates" ]; then
    echo "No duplicate lines found in ${scp}."
else
    echo "Duplicate lines found in ${scp}:"
    echo "$duplicates"
    exit 1
fi
