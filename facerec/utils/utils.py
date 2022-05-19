import os
import re
import json

import numpy as np
from PIL import Image

def get_embedding(model, face_pixels):
    """Get face embedding for one face.
    """
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)[0]
    # Perform L2-normalization as according to facenet paper
    # (make sure output is on the hypersphere with radius 1.0)
    yhat /= np.sqrt(np.sum(yhat ** 2))

    return yhat

def load_images_map(images_dir, features_dir = None):
    """From all face images, produce an easy lookup table.

    Format: {frame_index1: set(bbs_tuple1, bbs_tuple2, etc...)}
    """
    _, _, files = next(os.walk(images_dir))

    # facerec image file format: <movie_id>:<frame_i>:x1_y1_x2_y2.jpeg
    image_map = {}
    for name in files:
        name, ext = os.path.splitext(os.path.basename(name))
        if ext != ".jpeg":
            continue
        _, frame_str, box_str = name.split(":")
        frame_i = int(frame_str)
        bbox = tuple([int(p) for p in box_str.split("_")])
        if frame_i not in image_map:
            image_map[frame_i] = set()
        image_map[frame_i].add(bbox)

    if len(image_map)==0 and features_dir is not None:
        _, _, files = next(os.walk(features_dir))
        for name in files:
            _, ext = os.path.splitext(os.path.basename(name))
            if ext != ".jsonl":
                continue
            for l in open(features_dir+'/'+name):
                s = json.loads(l)
                frame_i = s["frame"]
                bbox = tuple(s["box"])
                if frame_i not in image_map:
                    image_map[frame_i] = set()
                image_map[frame_i].add(bbox)

    return image_map

def read_features(data_dir: str):
    features_file = os.path.join(data_dir, "features.jsonl")
    vector_map = {}
    with open(features_file, "r") as file:
        for line in file:
            obj = json.loads(line)
            frame, box = obj["frame"], tuple(obj["box"])
            vector = np.array(obj["embedding"], dtype=np.float32)
            if frame not in vector_map:
                vector_map[frame] = {}
            vector_map[frame][box] = vector
    # Map frame_index, box -> vector
    return vector_map

def get_vectors(trajectory, vector_map):
    """Read out all existing embedding vectors from a trajectory.

    Args:
        trajectory: Dict - from trajectories file.
        vector_map: From read_features.
    """
    vectors = []
    for frame, bbs in enumerate(trajectory["bbs"], start=trajectory["start"]):
        tup_bbs = tuple(bbs)
        if frame in vector_map and tup_bbs in vector_map[frame]:
            vectors.append(vector_map[frame][tup_bbs])
    return np.array(vectors, dtype=np.float32)
