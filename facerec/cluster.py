#! /usr/bin/env python3

import os
import json
import argparse
import glob
from typing import List, Set

import scipy.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import read_features, get_vectors

def read_trajectories(data_dir: str, vector_map):
    trajectories_file = os.path.join(data_dir, "trajectories.jsonl")
    trajectories = []
    mean_embeddings = []
    with open(trajectories_file, "r") as file:
        for line in file:
            trajectory = json.loads(line)
            vectors = get_vectors(trajectory, vector_map)
            trajectories.append(trajectory)
            # TODO: something better than mean? Eg. prefer front faces.
            mean_embeddings.append(vectors.mean(axis=0))
    mean_embeddings = np.array(mean_embeddings)
    return trajectories, mean_embeddings

def cluster_once(vectors, n_clusters):
    #print(vectors, type(vectors), vectors.shape)
    if vectors.shape[0]==0:
        return np.array([], dtype=np.int32)
    if vectors.shape[0]==1:
        return np.array([1], dtype=np.int32)
    link = cluster.hierarchy.linkage(vectors, method="complete")
    clusters = cluster.hierarchy.fcluster(link, t=n_clusters, criterion="maxclust")
    #print(clusters, type(clusters), clusters.dtype)
    return clusters

def split_and_merge(clusters, min_size=20, max_size=40):
    """Split clusters into new clusters that are no greater than max size and no
    smaller then min_size.

    Note: min_size is not strictly guaranteed!
    """
    new_clusters = np.zeros(clusters.size, dtype=np.int32)
    next_cluster_id = 0

    # 1. Split clusters that were too big
    cluster_ids, counts = np.unique(clusters, return_counts=True)
    for ci, n in zip(cluster_ids, counts):
        idx = np.where(ci == clusters)[0]
        if n > max_size:
            n_splits = (n + max_size - 1) // max_size
            split_size = (n + n_splits - 1) // n_splits
            for i in range(n_splits):
                new_clusters[idx[i * split_size:(i + 1) * split_size]] = next_cluster_id
                next_cluster_id += 1
        else:
            new_clusters[idx] = next_cluster_id
            next_cluster_id += 1

    # 2. Merge clusters that were too small
    # Note: generally in this function we'll consider all samples to be sufficient
    # to be in a single cluster, so we can merge any way we like.
    cluster_ids, counts = np.unique(new_clusters, return_counts=True)
    too_small_idx = np.where(counts < min_size)[0]
    cluster_ids = cluster_ids[too_small_idx]
    counts = counts[too_small_idx]

    prev_i = 0
    for upper_i in range(1, too_small_idx.size):
        bundle_sum = counts[prev_i:upper_i].sum()
        if bundle_sum >= min_size or upper_i == too_small_idx.size - 1:
            # Set all of these clusters to have the id of the first one
            idx = np.isin(new_clusters, cluster_ids[prev_i:upper_i])
            new_clusters[idx] = cluster_ids[prev_i]
            prev_i = upper_i

    return new_clusters

def relabel(clusters):
    """Make sure clusters are labeled from 0 strictly. Basically LabelEncoder.
    """
    new_clusters = np.zeros(clusters.size, dtype=np.int32)
    for i, ci in enumerate(np.unique(clusters)):
        new_clusters[np.where(ci == clusters)[0]] = i
    return new_clusters

def cluster_trajectories(trajectories, embeddings, size, min_size, max_size):
    """Perform clustering, while keeping a max cluster size for ease of use
    when labeling later on.

    Args:
        trajectories: trajectory objects that we want to cluster, using their
            embeddings.
        embeddings: Face embeddings corresponding to each trajectory; vectors
            that we will cluster.
        size: Preferred size. Ideally we'd like the clusters to be this size.
        min_size: Preferred minimum size - not strictly guaranteed since we'd
            rather not have outliers mix with the rest.
        max_size: Maximum size of clusters.
    """
    N = len(trajectories)
    n_clusters = N // size

    # Perform initial clustering
    clusters = cluster_once(embeddings, n_clusters)
    cluster_ids, counts = np.unique(clusters, return_counts=True)

    # Split clusters that were too big
    # Note: on this level, we won't merge clusters that are too small.
    for ci, n in zip(cluster_ids, counts):
        if n > max_size:
            n_splits = (n + max_size - 1) // max_size
            idx = np.where(ci == clusters)[0]
            new_clusters = cluster_once(embeddings[idx], n_splits)
            new_clusters = split_and_merge(new_clusters, min_size, max_size)

            next_cluster_id = clusters.max() + 1
            clusters[idx] = next_cluster_id + new_clusters

    clusters = relabel(clusters)
    cluster_ids, counts = np.unique(clusters, return_counts=True)
    print(f"Number of clusters: {len(cluster_ids)}")

    # Finally, relabel clusters with those with most trajectories first
    # So cluster 0 is the one with most trajectories in it, etc.
    order = np.argsort(np.argsort(-counts))[cluster_ids]
    clusters = order[clusters]

    return clusters

def write_clusters(clusters: np.array, data_dir: str, movie_id: int):
    out_file = os.path.join(data_dir, "clusters.json")
    with open(out_file, "w") as file:
        json.dump(
            {"clusters": [int(c) for c in clusters], "movie_id": movie_id},
            file, indent=None, separators=(",", ":")
        )
        file.write("\n")

    print(f"Wrote trajectory clusters to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--size", type=int, default=18,
                        help="Preferred size of output clusters (in trajectory count).")
    parser.add_argument("--min-size", type=int, default=12,
                        help="Preferred minimum size of output clusters.")
    parser.add_argument("--max-size", type=int, default=24,
                        help="Maximum size of output clusters.")
    parser.add_argument("--path", type=str, default=".")
    args = parser.parse_args()

    data_dirs = glob.glob(args.path)
    for data_dir in data_dirs:
        # data_dir will be a movie directory like ./data/123456-data
        data_dir = data_dir.rstrip("/")
        movie_id = int(os.path.basename(data_dir).split("-")[0])
        print(f"Clustering: {data_dir}")

        vector_map = read_features(data_dir)
        trajectories, mean_embeddings = read_trajectories(data_dir, vector_map)

        clusters = cluster_trajectories(
            trajectories, mean_embeddings, args.size, args.min_size, args.max_size
        )

        write_clusters(clusters, data_dir, movie_id)
        print()
