#!/bin/bash

# Define the simulations, dimensions, and data configurations
declare -a sims=("r1" "r2" "spring" "charge")
declare -a dims=("2" "3")
declare -a phases=("train" "val" "test")
declare -a seeds=("1" "2" "3")
declare -a ns_values=("7500" "2500" "10000") # Train, Val, Test

# Base directory for data (modify as per actual structure)
base_dir="data/test_data_hpc_reproduce"

for sim in "${sims[@]}"; do
    for dim in "${dims[@]}"; do
        # Set the number of bodies based on dimension
        if [ "$dim" -eq "2" ]; then
            n_bodies="4"
        elif [ "$dim" -eq "3" ]; then
            n_bodies="8"
        else
            echo "Unsupported dimension: $dim"
            continue
        fi

        for i in "${!phases[@]}"; do
            phase=${phases[$i]}
            seed=${seeds[$i]}
            ns=${ns_values[$i]}

            # Define the directory path for current setup
            dir_path="${base_dir}/${sim}_${dim}d/${phase}/raw"

            # Ensure directory exists
            mkdir -p "${dir_path}"

            # Command to run simulation
            echo "Running simulation for $sim in $dim dimensions with $n_bodies bodies for $phase phase..."
            cmd="python simulations/run_sims.py $sim $dir_path --dim $dim --nt 500 --ns $ns --n $n_bodies --seed $seed"
            echo $cmd
            eval $cmd

            # Check if the command was successful
            if [ $? -eq 0 ]; then
                echo "Simulation data saved to: $dir_path"
            else
                echo "Failed to run simulation for $sim in $dim dimensions for $phase phase."
                exit 1
            fi
        done
    done
done

echo "All simulations completed successfully."
