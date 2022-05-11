# facerec

Collection of scripts for face recognition of actors in films. The scripts produce a rigid data format that is consumed by the backend of the labeling system: https://github.com/ekreutz/video-labeler.

## General pipeline

1. Detect faces (bounding boxes) using RetinaFace
2. Track detected faces to form trajectories (sequential bounding boxes over time)
3. Perform shot segmentation
4. Perform feature extraction on faces (embedddings, using FaceNet)
5. Cluster trajectories using feature vectors
6. Probabilistic classification on the cluster-level using KNN

## Scripts

The scripts are run in the following order:

### `facerec/extract.py`

Performs the first 4 steps of the pipeline. Extracts and saves a lot of data from the raw video/movie file. The script is meant to be run in a CPU cluster that uses [SLURM](https://docs.csc.fi/computing/running/getting-started/). This enabled sharding of jobs, so that a movie can be processed in parallel.

Example run (shard index 0 out of 256 total):

```
# In this case, "12345" is the unique id for the movie "Cool Movie Film".
python -u ./facerec/extract.py \
    --shard 0 \
    --n 256 \
    --out /output/folder/root/path/data \
    ./films/12345-CoolMovieFilm.mp4
```

The script creates a movie-specific folder: `12345-data` within the specified `out` path.

### `facerec/merge_shards.py`

Merges all shards that were created by a parallel run of `extract.py`.

Example run:

```
python -u ./facerec/merge_shards.py \
    --path /output/folder/root/path/12345-data
```

The script creates top-level `trajectories.jsonl`, `features.jsonl` and `scene_changes.json` files within the directory `12345-data`.

### `facerec/cluster.py`

Perform clustering on extracted features.

Example run:

```
python -u ./facerec/cluster.py \
    --path /output/folder/root/path/12345-data
```

Produces a file `clusters.json` within `12345-data`.

### `facerec/classify_knn.py`

Perform classification on a cluster level. These predictions are used as initial guesses in the annotation tool. This script requires the presence of some additional files/features to work. These files are pre-computed embeddings for database images of the actors; precomputed embedding vectors and other metadata.

Put the files (download sample [here](https://drive.google.com/file/d/1GFOCDFkrTsMBV29Uwpioq6JFrcgQY0X1/view?usp=sharing)) into the root path of facerec, from where you want to run `facerec/classify_knn.py`. The files have the names `actors.csv`, `actor_images.csv`, `actor_embeddings.jsonl`.

Example run:

```
python -u ./facerec/classify_knn.py \
    --path /output/folder/root/path/12345-data
```

Produces a file `predictions.json` within `12345-data`.
