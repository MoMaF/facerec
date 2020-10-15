import os
import sys
import argparse
import json
from collections import namedtuple
from time import time

import cv2
import numpy as np
import tensorflow
from PIL import Image, ImageDraw

import utils.utils as utils
from detector import MTCNNDetector, RetinaFaceDetector
from sort import Sort

CROP_MARGIN = 0
FACE_IMAGE_SIZE = 160  # save face crops in this image resolution! (required!)

Options = namedtuple(
    "Options",
    ["out_path", "n_shards", "shard_i", "save_every", "min_trajectory", "max_trajectory_age"],
)

def bbox_float_to_int(bbox_float, max_w, max_h):
    bbox_float = np.maximum(np.round(bbox_float), 0)
    bbox_float[2] = np.minimum(bbox_float[2], max_w)
    bbox_float[3] = np.minimum(bbox_float[3], max_h)
    return [int(c) for c in bbox_float]

def save_trajectories(file, trackers, video_w, video_h):
    """Save trajectories from all given trackers, to file.
    """
    # Extract trajectories from trackers and write to jsonl
    for trk in trackers:
        trajectory = []
        detected = []
        for bbox_float, d in trk.history:
            bbox_int = bbox_float_to_int(bbox_float, video_w, video_h)
            trajectory.append(bbox_int)
            detected.append(d)

        out_obj = {
            "start": trk.first_frame,
            "len": len(trajectory),
            "bbs": trajectory,
            "detected": detected,
        }
        json.dump(out_obj, file, indent=None, separators=(",", ":"))
        file.write("\n")

    return len(trackers)

def process_frame(frame_data, video_w, video_h, features_file, images_dir, min_trajectory_len):
    """Save faces + features from a frame, and creating face embeddings.
    """
    # Filter to faces with a valid trajectory (len > MIN)
    valid_faces = [
        face for face in frame_data["faces"]
        if multi_tracker.has_valid_tracker(face["detection_id"])
    ]

    img = Image.fromarray(frame_data["img_np"])
    for face in valid_faces:
        # Retrieve the bbox filtered by the Kalman filter, instead of actual detection
        filtered_box = multi_tracker.get_detection_bbox(face["detection_id"])
        filtered_box = bbox_float_to_int(filtered_box, video_w, video_h)

        # Crop onto face only
        cropped = img.crop(tuple(filtered_box))
        resized = cropped.resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)

        # Get face embedding vector via facenet model
        scaledx = np.array(resized)
        scaledx = scaledx.reshape(-1, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE, 3)
        embedding = utils.get_embedding(facenet, scaledx[0])

        # Save face image and features
        box_label = frame_data["label"] + "_{}_{}_{}_{}".format(*filtered_box)
        resized.save(f"{images_dir}/kept-{box_label}.jpeg")
        json.dump({
            "frame": frame_data["index"],
            "label": box_label,
            "embedding": embedding.tolist(),
            "box": filtered_box,
            "keypoints": face["keypoints"],
        }, features_file, indent=None, separators=(",", ":"))
        features_file.write("\n")

    return len(valid_faces)

def process_video(file, opt: Options):
    """Process entire video and extract face boxes.
    """
    assert opt.shard_i < opt.n_shards and opt.shard_i >= 0, "Bad shard index."

    cap = cv2.VideoCapture(file)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    shard_len = (n_total_frames + opt.n_shards - 1) // opt.n_shards
    beg = shard_len * opt.shard_i
    end = min(beg + shard_len, n_total_frames)  # not inclusive
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, beg), \
        f"Couldn't set start frame to: {beg}"

    # We'll write (face) images, features and trajectories to disk
    label, _ = os.path.splitext(os.path.basename(file))
    features_dir = f"{opt.out_path}/{label}-data/features"
    trajectories_dir = f"{opt.out_path}/{label}-data/trajectories"
    images_dir = f"{opt.out_path}/{label}-data/images"
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    features_path = f"{features_dir}/features_{label}_{beg}-{end}.jsonl"
    features_file = open(features_path, mode="w")
    trajectories_path = f"{trajectories_dir}/trajectories_{label}_{beg}-{end}.jsonl"
    trajectories_file = open(trajectories_path, mode="w")

    print(f"Movie file: {os.path.basename(file)}")
    print(f"Total length: {(n_total_frames / fps / 3600):.1f}h ({fps} fps)")
    print(f"Shard {(opt.shard_i + 1)} / {opt.n_shards}, len: {shard_len} frames")
    print(f"Processing frames: {beg} - {end} (max: {n_total_frames})")

    buf = []
    saved_frames_count = 0
    saved_boxes_count = 0
    saved_traj_count = 0

    for f in range(beg, end):
        ret, frame = cap.read()

        if not ret:
            break

        frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect(frame_img)
        buf.append({
            "index": f,
            "img_np": frame_img,
            "faces": faces,
            "label": label + ":" + str(f).zfill(6),
        })

        # Let the tracker know of new detections
        detections = np.array([[*f["box"], 0.95] for f in faces]).reshape((-1, 5))
        detection_ids = multi_tracker.update(detections, frame=f)
        # Make each face keep a reference to its tracker
        for i, face in enumerate(faces):
            face["detection_id"] = detection_ids[i]

        # Clean up expired trajectories (-> save to file)
        expired_tracks = multi_tracker.pop_expired(expiry_age=2 * opt.min_trajectory)
        saved_traj_count += save_trajectories(trajectories_file, expired_tracks, video_w, video_h)

        # Extract good face boxes from middle frame, save those
        if len(buf) == opt.min_trajectory:
            frame_data = buf.pop(0)
            if frame_data["index"] % opt.save_every == 0:
                n_saved_faces = process_frame(
                    frame_data, video_w, video_h, features_file, images_dir, opt.min_trajectory
                )
                saved_boxes_count += n_saved_faces
                saved_frames_count += int(n_saved_faces > 0)

    # Save remaining frames and trajectories
    for frame_data in buf:
        if frame_data["index"] % opt.save_every == 0:
            n_saved_faces = process_frame(
                frame_data, video_w, video_h, features_file, images_dir, opt.min_trajectory
            )
            saved_boxes_count += n_saved_faces
            saved_frames_count += int(n_saved_faces > 0)

    expired_tracks = multi_tracker.pop_expired(expiry_age=0)
    saved_traj_count += save_trajectories(trajectories_file, expired_tracks, video_w, video_h)

    features_file.close()
    trajectories_file.close()
    cap.release()
    print(f"Saved {saved_boxes_count} boxes from {saved_frames_count} different frames")
    print(f"and {saved_traj_count} trajectories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--n-shards", type=int, default=256)
    parser.add_argument("--shard-i", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out-path", type=str, default="./data")
    parser.add_argument("--min-trajectory", type=int, default=5)
    parser.add_argument("--max-trajectory-age", type=int, default=10)
    parser.add_argument("file")
    args = parser.parse_args()

    start_time = time()

    # Models found at: https://github.com/D2KLab/Face-Celebrity-Recognition
    facenet = tensorflow.keras.models.load_model("models/facenet_keras.h5")
    facenet.load_weights("models/facenet_keras_weights.h5")

    # Comment out 1, same wrapped api!
    # detector = MTCNNDetector()
    detector = RetinaFaceDetector()

    # Tracker - SORT. Has nothing to do with sorting
    multi_tracker = Sort(
        max_age=args.max_trajectory_age,
        min_hits=args.min_trajectory,
        iou_threshold=args.iou_threshold,
    )

    _, ext = os.path.splitext(os.path.basename(args.file))
    if ext in [".mpeg", ".mpg", ".mp4", ".avi", ".wmv"]:
        options = Options(
            n_shards=args.n_shards,
            shard_i=args.shard_i,
            save_every=args.save_every,
            out_path=args.out_path,
            max_trajectory_age=args.max_trajectory_age,
            min_trajectory=args.min_trajectory,
        )
        process_video(args.file, options)

        minutes, seconds = divmod(time() - start_time, 60)
        print(f"Completed in {int(minutes)} minutes, {int(seconds)} seconds.")