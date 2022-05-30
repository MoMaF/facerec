#! /bin/bash -l

#SBATCH --job-name=face_extract
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --partition=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --array=0-255
#SBATCH --exclude=./exclude-list.txt

if [[ $# == 0 ]]; then
    echo $0 : video file name argument missing
    exit 1
fi

if [[ $# > 1 ]]; then
    echo $0 : too many arguments
    exit 1
fi

echo Running in `hostname` $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $1

. ./venv/bin/activate

python3 -u ./facerec/extract.py \
    --n-shards $SLURM_ARRAY_TASK_COUNT \
    --shard-i $SLURM_ARRAY_TASK_ID \
    --save-every 1 \
    --out-path out \
    --no-images \
    $1

if [[ $? -ne 0 ]]
then
    echo FAILED in `hostname` $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $1
    exit 1
fi

echo SUCCESS $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $1

seff $SLURM_JOB_ID
