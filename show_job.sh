#!/bin/bash

# Check if a job ID was passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <job_id>"
    exit 1
fi

# Assign the job ID from the first argument
JOBID=$1

echo "Monitoring SLURM job: $JOBID"

# Loop to check the job status every 15 seconds
while true; do
    # Check job status
    STATUS=$(scontrol show job $JOBID | grep "JobState")

    echo $STATUS

    # Break the loop if job is completed or failed
    if [[ $STATUS =~ "COMPLETED" ]] || [[ $STATUS =~ "FAILED" ]] || [[ $STATUS =~ "CANCELLED" ]]; then
        echo "Job $JOBID has finished with status $STATUS"
        break
    fi

    # Wait for 15 seconds before checking again
    sleep 15
done
