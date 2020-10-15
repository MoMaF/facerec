#!/bin/bash -l

#SBATCH --job-name=face_extract_job
#SBATCH --output=logs/array_job_out_%A_%a.txt
#SBATCH --error=logs/array_job_err_%A_%a.txt
#SBATCH --account=project_2002528
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-255

if [[ $# == 0 ]]; then
    echo $0 : video file name argument missing
    exit 1
fi

if [[ $# > 1 ]]; then
    echo $0 : too many arguments
    exit 1
fi

. ./venv/bin/activate

python -u ./extract.py \
    --n-shards 256 \
    --shard-i $SLURM_ARRAY_TASK_ID \
    --out-path /scratch/project_2002528/emil \
    $1
