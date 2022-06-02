#! /usr/bin/env python3

import os
import json
import argparse
import glob

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

from utils.utils import read_features, get_vectors


ACTOR_ID_PREFIX = "momaf:elonet_henkilo_"
ACTORS_PATH = "./actors.csv"
ACTOR_IMAGES_PATH = "./actor_images.csv"
ACTOR_EMBEDDINGS_PATH = "./actor_embeddings.jsonl"

def read_actors(path):
    """Read a mapping from movie to actors.
    """
    df = pd.read_csv(path, index_col="movie_id")
    return df

def read_actor_images(path):
    """Read csv that links images to actors.
    """
    df = pd.read_csv(path)

    # Find images with only 1 actor known in them
    files_df = df.groupby(by="filename").count()
    files = files_df[files_df.actor_id == 1].index

    df = df[df.filename.isin(files)]
    return df

def read_actor_embeddings(path, actor_images_df):
    # Read embeddings from images that had only 1 detection
    valid_files = set(actor_images_df.filename)
    embeddings = {}
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["image"] not in valid_files or data["n"] != 1:
                continue
            embeddings[data["image"]] = data["faces"][0]["vector"]

    # Dict: filename -> embedding vector
    return embeddings

def get_class_data(data_dir, actor_df, actor_images_df, embeddings, min_samples=20):
    movie_id = int(os.path.basename(data_dir).split("-")[0])

    # Find actors for current movie
    current_actors = set(actor_df.loc[movie_id].id)
    assert len(current_actors) > 0, "No actors found for movie?"

    # Find subset of actor images and embeddings, for this movie
    actor_images_df = actor_images_df[actor_images_df.filename.isin(embeddings)]
    actor_images_df = actor_images_df[actor_images_df.actor_id.isin(current_actors)]
    embedding_dim = len(embeddings[list(embeddings.keys())[0]])

    if len(actor_images_df) == 0:
        # Return empty classifications
        return np.empty((0, embedding_dim), dtype=np.float32), np.empty((0,), dtype=np.int32)

    round_actors, counts = np.unique(actor_images_df.actor_id, return_counts=True)
    n_actors = len(round_actors)
    n_samples = max(np.min(counts), min_samples)

    print(f"Movie had {len(current_actors)} actors, round has {n_actors}.")

    X = np.zeros((n_actors * n_samples, embedding_dim), dtype=np.float32)
    y = np.zeros(n_actors * n_samples, dtype=np.int32)
    for i, (actor_id, row_df) in enumerate(actor_images_df.groupby("actor_id")):
        vectors = row_df.filename.apply(embeddings.get).tolist()

        # Upsample if we had less than `n_samples` samples.
        m = len(vectors)
        multiplier = (n_samples + m - 1) // m

        vectors = np.array((vectors * multiplier)[:n_samples], dtype=np.float32)
        assert vectors.shape[0] == n_samples

        start, end = i * n_samples, (i + 1) * n_samples
        X[start:end] = vectors
        y[start:end] = int(actor_id)

    return X, y

def classify(data_dir, X, y, save_p_higher=0.05):
    """Classify actors in individual images in each trajectory and cluster, then
    aggregate to get trajectory and cluster-level predictions.
    """
    movie_id = int(os.path.basename(data_dir).split("-")[0])

    trajectories_file = os.path.join(data_dir, "trajectories.jsonl")
    clusters_file = os.path.join(data_dir, "clusters.json")
    predictions_file = os.path.join(data_dir, "predictions.json")

    feature_vector_map = read_features(data_dir)

    # Read clusters
    with open(clusters_file, "r") as file:
        clusters = np.array(json.load(file)["clusters"])
    uniq_clusters = sorted(set(clusters))

    # Note: even with more than 3 classes, the predictions could still be off.
    if len(np.unique(y)) < 3 or len(X) == 0:
        with open(predictions_file, "w") as file:
            json.dump({int(ci): {} for ci in uniq_clusters}, file)
        print(f"Not enough actor data. Wrote empty predictions: {predictions_file}")
        return

    # Initialize classifier
    dims = X.shape[1]
    knn = KNeighborsClassifier(n_neighbors=10, weights="uniform").fit(X, y)
    classes = knn.classes_  # classes are actor ids

    # Get averaged predictions for each trajectory
    trajectory_preds = []
    with open(trajectories_file, "r") as file:
        for line in file:
            trajectory = json.loads(line)
            vectors = get_vectors(trajectory, feature_vector_map)

            if len(vectors) > 0:
                preds = knn.predict_proba(vectors)
                mean_pred = preds.mean(axis=0)
            else:
                mean_pred = np.zeros(dims, dtype=np.float32)

            trajectory_preds.append(mean_pred)
    trajectory_preds = np.array(trajectory_preds)
    assert len(trajectory_preds) == len(clusters)

    # Finally, average over clusters too.
    cluster_preds = {}
    for ci in uniq_clusters:
        idx = np.where(clusters == ci)[0]
        preds = trajectory_preds[idx]
        cluster_pred = preds.mean(axis=0)

        # Save only predictions higher than some minimum
        passed_min = np.where(cluster_pred > save_p_higher)[0]
        cluster_preds[int(ci)] = {
            f"{ACTOR_ID_PREFIX}{classes[i]}": round(float(cluster_pred[i]), 5) for i in passed_min
        }

    # Save predictions to file
    with open(predictions_file, "w") as file:
        obj = {
            "movie_id": movie_id,
            "predictions": cluster_preds,
        }
        json.dump(obj, file)

    print(f"Wrote predictions: {predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--path", type=str,
                        help="Path to data directory for a film.")
    args = parser.parse_args()

    # Read data about actors and actor images
    actors_df = read_actors(ACTORS_PATH)
    actor_images_df = read_actor_images(ACTOR_IMAGES_PATH)
    embeddings = read_actor_embeddings(ACTOR_EMBEDDINGS_PATH, actor_images_df)

    data_dirs = glob.glob(args.path)
    for data_dir in data_dirs:
        # data_dir will be a movie directory like ./data/123456-data
        data_dir = data_dir.rstrip("/")
        print(f"Predicting for {data_dir} using KNN...")

        X, y = get_class_data(
            data_dir, actors_df, actor_images_df, embeddings, min_samples=20,
        )
        classify(data_dir, X, y)
        print()
