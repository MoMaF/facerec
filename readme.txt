srun -p small -A project_462000139 --time=08:00:00 --mem 32G --pty bash -i
srun -p small-g --gpus-per-node=1 -A project_462000139 --time=08:00:00 --mem 32G --pty bash -i

export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
module use /appl/local/csc/modulefiles
module load tensorflow
export PYTHONUSERBASE=/scratch/project_462000189/jorma/momaf/github/facerec/python_base

pip install -r requirements.txt
pip install --no-deps retinaface

./facerec/extract.py --n-shards 100 --shard-i 50 ../../films/125261-PekkaJaPatkaPahassaPulassa-1955.mp4

python3 -u ./facerec/extract.py --n-shards 100 --shard-i 50 --save-every 1 --out-path out2 ../../films/125261-PekkaJaPatkaPahassaPulassa-1955.mp4

sbatch scripts/extract.sh ../../films/125261-PekkaJaPatkaPahassaPulassa-1955.mp4

./facerec/merge_shards.py --path out/125261-data

./facerec/cluster.py --path out/125261-data

./facerec/prepare-actors.py --film 125261 --path out/125261-data --actors-dir out/125261-data 

./facerec/classify_knn.py --path out/125261-data --actors-dir out/125261-data 

./facerec/cluster.py --path out/125261-data

./facerec/make_subtitles.py --path out/125261-data

./facerec/check_twins.py --path out/125261-data
