#! /usr/bin/env python3

import os
import json
import argparse
import glob
import zipfile
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.utils import read_features, get_vectors

emb_name = '20170512-110547'

ACTOR_ID_PREFIX = "momaf:elonet_henkilo_"
#ACTORS_PATH = "./actors.csv"
#ACTOR_IMAGES_PATH = "./actor_images.csv"
#ACTOR_EMBEDDINGS_PATH = "./actor_embeddings.jsonl"

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

actor_names = {}

def read_actor_embeddings(zipf):
    embeddings = []
    z = zipfile.ZipFile(zipf)
    for j in z.namelist():
        if j[-5:]=='.json':
            d = json.loads(z.read(j))
            #print(d.keys())
            if 'box' in d:
                aid = int(d['actorID'])
                #print(d['actorname'], d['actorID'], d['filmname'], d['filename'])
                embeddings.append((aid, d['embeddings'][emb_name]))
                actor_names[aid] = d['actorname']
    return embeddings

def read_actor_embeddings_old(path, actor_images_df):
    # Read embeddings from images that had only 1 detection
    valid_files = set(actor_images_df.filename)
    embeddings = {}
    n_total = 0
    n_valid = 0
    dim = 0
    with open(path, "r") as file:
        for line in file:
            n_total += 1
            data = json.loads(line)
            if data["image"] not in valid_files or data["n"] != 1:
                continue
            vec = data["faces"][0]["vector"]
            dim = len(vec)
            embeddings[data["image"]] = vec
            n_valid += 1
    print(f'Read {len(embeddings)} face embeddings of dimensionality {dim} from {path} \n'+
          f'that contained {n_valid} valid single-face images from total of {n_total} images')
    # Dict: filename -> embedding vector
    return embeddings

def actor_name(t):
    return actor_names[t]

def actor_name_old(t, ac):
    return ac[ac['id']==t].iloc[0]['name']

def summarize_embeddings(embeddings, actor_df, actor_images_df):
    stat = {}
    for i in embeddings.keys():
        #print(i)
        a = actor_images_df[actor_images_df['filename']==i]
        assert a.shape[0]==1
        #print(a, a.shape)
        aid = a.iloc[0]['actor_id']
        if aid not in stat:
            stat[aid] = 0
        stat[aid] += 1
    l = sorted([ [j, i] for i, j in stat.items() ], reverse=True)
    #print(l)
    n = 0
    for i, j in l:
        print(f'{j:7} {actor_name(j, actor_df):25} {i:4}')
        n += i
    print(f'Total {n} embeddings of {len(l)} actors')
        
def get_class_data(embeddings, min_samples=20):
    if len(embeddings) == 0:
        # Return empty classifications
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)

    actors = {}
    for a, v in embeddings:
        embedding_dim = len(v)
        if a not in actors:
            actors[a] = []
        actors[a].append(v)
    #print(actors)
    min_actors, max_actors = (0, 0)
    for _, vv in actors.items():
        n = len(vv)
        if min_actors==0 or n<min_actors:
            min_actors = n
        if n>max_actors:
            max_actors = n

    n_actors = len(actors)
    n_samples = max(min_actors, min_samples)

    print(f"Images for {n_actors} actors (min {min_actors}, max {max_actors}) {n_samples} will be sampled.")

    X = np.zeros((n_actors * n_samples, embedding_dim), dtype=np.float32)
    y = np.zeros(n_actors * n_samples, dtype=np.int32)
    for i, actor_id in enumerate(actors.keys()):
        vectors = actors[actor_id]
        # Upsample if we had less than `n_samples` samples.
        m = len(vectors)
        print(f'{actor_id:7} {actor_name(actor_id):25} {m:3} -> {n_samples:3}')
        multiplier = (n_samples + m - 1) // m

        vectors = np.array((vectors * multiplier)[:n_samples], dtype=np.float32)
        assert vectors.shape[0] == n_samples

        start, end = i * n_samples, (i + 1) * n_samples
        X[start:end] = vectors
        y[start:end] = actor_id
    print(f'Total {y.shape[0]} vectors')
    
    return X, y

def get_class_data_old(data_dir, actor_df, actor_images_df, embeddings, min_samples=20):
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
        print(f'{actor_id:7} {actor_name(actor_id, actor_df):25} {m:3} -> {n_samples:3}')
        multiplier = (n_samples + m - 1) // m

        vectors = np.array((vectors * multiplier)[:n_samples], dtype=np.float32)
        assert vectors.shape[0] == n_samples

        start, end = i * n_samples, (i + 1) * n_samples
        X[start:end] = vectors
        y[start:end] = int(actor_id)
    print(f'Total {y.shape[0]} vectors')
    
    return X, y

def classify(data_dir, X, y, k, save_p_higher=0.05):
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
    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X, y)
    classes = knn.classes_  # classes are actor ids

    # Get averaged predictions for each trajectory
    trajectory_preds = []
    first = True
    with open(trajectories_file, "r") as file:
        for line in file:
            trajectory = json.loads(line)
            vectors = get_vectors(trajectory, feature_vector_map, emb_name)
            if first:
                print(f'Embeddings in {trajectories_file} are {len(vectors[0])} dimensional')
                first = False
            
            if len(vectors) > 0:
                preds = knn.predict_proba(vectors)
                mean_pred = preds.mean(axis=0)
            else:
                mean_pred = np.zeros(dims, dtype=np.float32)

            trajectory_preds.append(mean_pred)
    trajectory_preds = np.array(trajectory_preds)
    assert len(trajectory_preds) == len(clusters)
    print(trajectory_preds.shape)
    
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
    parser = argparse.ArgumentParser(allow_abbrev=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--path", type=str, default=".",
                        help="path to JSON data directory for a film")
    parser.add_argument('--actors-dir', type=str, default='.',
                        help='directory to find actor-images.zip')
    args = parser.parse_args()

    # Read data about actors and actor images
    #actors_df = read_actors(ACTORS_PATH)
    #actor_images_df = read_actor_images(ACTOR_IMAGES_PATH)
    #embeddings = read_actor_embeddings(ACTOR_EMBEDDINGS_PATH, actor_images_df)
    embeddings = read_actor_embeddings(args.actors_dir+'/actor-images.zip')
    #keys = list(embeddings.keys())
    #print(f'Read {len(keys)} face embeddings with dimensionality {len(embeddings[keys[0]])}'+
    #      f' from {ACTOR_EMBEDDINGS_PATH}')

    # summarize_embeddings(embeddings, actors_df, actor_images_df)
    
    data_dirs = glob.glob(args.path)
    for data_dir in data_dirs:
        # data_dir will be a movie directory like ./data/123456-data
        data_dir = data_dir.rstrip("/")
        min_samples = 20
        k = 10
        print(f'Predicting for {data_dir} using k-NN with k={k} and min_samples={min_samples}')

        X, y = get_class_data(embeddings, min_samples)
        #X, y = get_class_data_old(
        #    data_dir, actors_df, actor_images_df, embeddings, min_samples=20,
        #)
        classify(data_dir, X, y, k)
        print()
        break
    
