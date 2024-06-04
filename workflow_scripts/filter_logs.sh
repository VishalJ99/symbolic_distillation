#!/bin/bash

# Directory containing log files
log_dir="$1"

# Loop through all .log files in the specified directory
for log_file in "$log_dir"/*.out; do
    # Check if the file contains the word "Traceback"
    if grep -q "Traceback" "$log_file"; then
        echo "Traceback found in: $log_file"
    fi
done
