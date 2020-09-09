#! /bin/bash

#SBATCH --time=4:00:00
#SBATCH --mem=20G

beg=0
end=-1

if [[ $# == 0 ]]; then
    echo $0 : video file name argument missing
    exit 1
fi

if [[ $# > 3 ]]; then
    echo $0 : too many arguments
    exit 1
fi

if [[ $# == 3 ]]; then
    beg=$2
    end=$3
fi

if [[ $# == 2 && X$SLURM_ARRAY_TASK_ID != X ]]; then
    beg=$(dc -e "$2 $SLURM_ARRAY_TASK_ID * p") 
    end=$(dc -e "$beg $2 + p")
fi

. venv-triton/bin/activate

./extract.py $1 $beg $end

