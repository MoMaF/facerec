#! /usr/bin/env python3
import os

import scipy.cluster as cluster
import matplotlib.pyplot as plt
import numpy as np


def read_features(file_path):
    """Read a single features file from extract.py.
    """
    labels = []
    vectors = []
    with open(file_path, "r") as f:
        for line in f:
            arr = line.split(",")
            frame_label, box_label = arr[:2]
            v = np.array([float(s) for s in arr[2:]])
            labels.append(box_label)
            vectors.append(v)
    return labels, vectors

def read_all_features(features_dir):
    _, _, files = next(os.walk(features_dir))
    labels = []
    vectors = []
    for features_file in files:
        l, v = read_features(os.path.join(features_dir, features_file))
        labels += l
        vectors += v
    return np.array(labels), np.array(vectors)

def cluster_vectors(vectors, shuffle_seed=42):
    np.random.seed(shuffle_seed)
    perm = np.random.permutation(vectors.shape[0]
    permuted_vectors = vectors[perm]

    link = cluster.hierarchy.linkage(permuted_vectors, method="complete")
    clusters = cluster.hierarchy.fcluster(link, t=100, criterion="maxclust")

    # Reverse permutation before returning
    perm_r = np.argsort(perm)
    return clusters[perm_r]


if __name__ == "__main__":
    labels, vectors = read_all_features("./features")
    clusters = cluster_vectors(vectors)

    # Store distances to cluster midpoints
    dists = np.zeros_like(clusters, dtype=np.float32)
    for cluster_i in np.unique(clusters):
        mask = clusters == cluster_i
        member_vecs = vectors[mask]

        mu = np.mean(member_vecs, axis=0)
        d = np.linalg.norm(member_vecs - mu, axis=1)
        dists[mask] = d
