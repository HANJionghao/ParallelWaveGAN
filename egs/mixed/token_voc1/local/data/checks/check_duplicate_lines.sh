#!/bin/bash

scp=$1
duplicates=$(awk '{count[$0]++} END {for (line in count) if (count[line] > 1) print count[line], line}' "$scp")

if [ -n "$duplicates" ]; then
    echo "Duplicate lines found in ${scp}:"
    echo "$duplicates"
    exit 1
fi
