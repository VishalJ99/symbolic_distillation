#!/bin/bash

# Base directory containing the YAML files
config_dir="$1"

# Loop over each YAML file found in the config_dir and its subdirectories
find "$config_dir" -type f -name "*.yaml" | while read -r yaml_file; do
    echo "Running command sbatch workflow_scripts/test_job.sh \"python\" \"src/test.py $yaml_file\""
    sbatch workflow_scripts/test_job.sh "python" "src/test.py $yaml_file"
done

echo "All jobs submitted."
