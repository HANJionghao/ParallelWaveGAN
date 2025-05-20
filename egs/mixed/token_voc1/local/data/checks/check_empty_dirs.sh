#!/bin/bash

root_dir=$1

if [ -z "$root_dir" ]; then
    echo "Usage: $0 <root_dir>"
    echo "Example: $0 /path/to/your/data"
    exit 1
fi

empty_dirs=$(find "$root_dir" -type d -empty)

if [ -n "$empty_dirs" ]; then
    echo "$root_dir contains empty directories:"
    echo "$empty_dirs"
    exit 1
fi