#!/bin/bash

# Base directory containing the YAML files
config_dir="configs/test_configs"

# Loop over each YAML file found in the config_dir and its subdirectories
find "$config_dir" -type f -name "*.yaml" | while read -r yaml_file; do
    echo "Running command sbatch hpc_scripts/test_job.sh \"python\" \"src/test.py $yaml_file\""
    sbatch hpc_scripts/test_job.sh "python" "src/test.py $yaml_file"
done

echo "All jobs submitted."
