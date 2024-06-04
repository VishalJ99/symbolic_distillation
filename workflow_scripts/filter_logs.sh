#!/bin/bash

# Directory containing log files
log_dir="$1"

# Loop through all .out files in the specified directory
for log_file in "$log_dir"/*.out; do
    # Check if the file contains the word "Traceback"
    if grep -q "Traceback" "$log_file"; then
        echo "Traceback found in: $log_file"
    else
        # If the file does not contain "Traceback", delete it
        echo "Deleting non-error log file: $log_file"
        rm "$log_file"
    fi
done
