#!/bin/bash

# Define the directory containing the .npy files
base_dir="../rds/hpc-work/test_data/"  # Modify this as needed to your base directory

# Find all .npy files in the directory
find "$base_dir" -type f -name "*.npy" | while read -r file; do
    echo "Checking file: $file"

    # Use Python to check for NaNs in the file
    nan_check=$(python -c "import numpy as np; arr = np.load('$file'); print(np.isnan(arr).any())")
    min_value=$(python -c "import numpy as np; arr = np.load('$file'); print(np.nanmin(arr) if np.isfinite(np.nanmin(arr)) else 'No valid min (all NaNs)')")
    max_value=$(python -c "import numpy as np; arr = np.load('$file'); print(np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 'No valid max (all NaNs)')")

    # Print min and max values
    echo "Min value: $min_value"
    echo "Max value: $max_value"

    # If NaNs are found, output a message and exit the script
    if [ "$nan_check" = "True" ]; then
        echo "Found NaNs in file: $file"
        exit 1
    fi
done
