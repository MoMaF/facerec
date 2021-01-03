import os
import sys
import argparse
import json
from collections import namedtuple
from time import time

import cv2
import numpy as np
import tensorflow
from PIL import Image, ImageOps

import utils.utils as utils
from detector import MTCNNDetector, RetinaFaceDetector
from sort import Sort
from scene import SceneChangeDetector

FACE_IMAGE_SIZE = 160  # save face crops in this image resolution! (required!)
SAVE_FACE_PADDING = 0.10  # before saving to disk, add this padding to show more face

Options = namedtuple(
    "Options",
    ["out_path", "n_shards", "shard_i", "save_every", "min_trajectory", "max_trajectory_age"],
)

def bbox_float_to_int(bbox_float, max_w, max_h, padding=0.0):
    """Convert float bounding box to integers.
    """
    bbox_float = np.array(bbox_float, dtype=np.float32)

    # Optionally pad to show more of the face
    dim = np.minimum(bbox_float[2] - bbox_float[0], bbox_float[3] - bbox_float[1])
    pad_px = padding * dim
    bbox_float += np.array([-pad_px, -pad_px, pad_px, pad_px])

    # Make sure box doesn't extend beyond image borders
    bbox_float = np.maximum(bbox_float, [0, 0, 0, 0])
    bbox_float = np.minimum(bbox_float, [max_w, max_h, max_w, max_h])
    bbox_float = np.round(bbox_float)

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
        # Retrieve the posterior bbox filtered by the Kalman filter
        filtered_box = multi_tracker.get_detection_bbox(face["detection_id"])

        # Crop onto face only (tight crop for embedding)
        tight_box = bbox_float_to_int(filtered_box, video_w, video_h)
        cropped = img.crop(tuple(tight_box))
        resized = cropped.resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)
        # Get face embedding vector via facenet model
        scaledx = np.array(resized)
        scaledx = scaledx.reshape(-1, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE, 3)
        embedding = utils.get_embedding(facenet, scaledx[0])

        # Produce padded crop that will be saved to disk (shown during annotation)
        padded_box = bbox_float_to_int(filtered_box, video_w, video_h, padding=SAVE_FACE_PADDING)
        padded_img = img.crop(tuple(padded_box))
        padded_img.thumbnail((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)

        # Determine if cropped image is actually grayscale. If so, convert.
        padded_a = np.array(padded_img).reshape((-1, 3))
        is_gray = np.all(padded_a[:,0] == padded_a[:,1])
        if is_gray:
            padded_img = ImageOps.grayscale(padded_img)

        # Save face image and features
        # Note: the box is named after the tight crop, even though the saved image
        # uses the padded box
        box_tag = frame_data["tag"] + ":{}_{}_{}_{}".format(*tight_box)
        padded_img.save(f"{images_dir}/{box_tag}.jpeg", quality=65)
        json.dump({
            "frame": frame_data["index"],
            "tag": box_tag,
            "embedding": embedding.tolist(),
            "box": tight_box,
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
    # For invalid video files, cap.get return 0.0. Use that as a validity check here.
    assert n_total_frames > 0, "Invalid video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    shard_len = (n_total_frames + opt.n_shards - 1) // opt.n_shards
    beg = shard_len * opt.shard_i
    end = min(beg + shard_len, n_total_frames)  # not inclusive
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, beg), \
        f"Couldn't set start frame to: {beg}"

    # We'll write (face) images, features, trajectories and scene changes to disk
    basename_no_ext, _ = os.path.splitext(os.path.basename(file))
    movie_id = int(basename_no_ext.split("-")[0])
    features_dir = f"{opt.out_path}/{movie_id}-data/features"
    trajectories_dir = f"{opt.out_path}/{movie_id}-data/trajectories"
    scene_changes_dir = f"{opt.out_path}/{movie_id}-data/scene_changes"
    images_dir = f"{opt.out_path}/{movie_id}-data/images"
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(scene_changes_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    features_path = f"{features_dir}/features_{movie_id}_{beg}-{end}.jsonl"
    features_file = open(features_path, mode="w")
    trajectories_path = f"{trajectories_dir}/trajectories_{movie_id}_{beg}-{end}.jsonl"
    trajectories_file = open(trajectories_path, mode="w")
    scene_changes_path = f"{scene_changes_dir}/scene_changes_{movie_id}_{beg}-{end}.json"

    scene = SceneChangeDetector(grayscale=False, crop=True, movie_id=movie_id)
    scene_changes = []

    print(f"Movie file: {os.path.basename(file)}")
    print(f"Total length: {(n_total_frames / fps / 3600):.1f}h ({fps} fps)")
    print(f"Shard {(opt.shard_i + 1)} / {opt.n_shards}, len: {shard_len} frames")
    print(f"Processing frames: {beg} - {end} (max: {n_total_frames})")

    buf = []
    saved_frames_count = 0
    saved_boxes_count = 0
    saved_traj_count = 0

    # Run past the end to allow easier merging with other shards
    end_with_overlap = min(end + opt.max_trajectory_age, n_total_frames)

    for f in range(beg, end_with_overlap):
        ret, frame = cap.read()

        if not ret:
            break

        frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scene_change_happened = scene.update(np.array(frame_img))

        faces = detector.detect(frame_img)
        buf.append({
            "index": f,
            "img_np": frame_img,
            "faces": faces,
            "tag": f"{movie_id}:{f}",
        })

        # Stop tracking previous trajectories (faces) if a scene change occurred
        if scene_change_happened:
            scene_changes.append(f)
            multi_tracker.kill_trackers()

        # Let the tracker know of new detections
        detections = np.array([[*f["box"], 0.95] for f in faces]).reshape((-1, 5))
        detection_ids = multi_tracker.update(detections, frame=f)
        # Assign the tracker's detection ids to each face.
        for i, face in enumerate(faces):
            face["detection_id"] = detection_ids[i]

        # Clean up expired trajectories (-> save to file)
        expired_tracks = multi_tracker.pop_expired(2 * opt.min_trajectory, f)
        saved_traj_count += save_trajectories(trajectories_file, expired_tracks, video_w, video_h)

        # For some frames, we save images + features
        if len(buf) == opt.min_trajectory and f < end:
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

    # Save scene changes to file
    with open(scene_changes_path, "w") as f:
        scene_changes = [f for f in scene_changes if f >= beg and f < end]
        json.dump({"frame_indices": scene_changes}, f, indent=None, separators=(",", ":"))

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
    parser.add_argument("--min-trajectory", type=int, default=3)
    parser.add_argument("--max-trajectory-age", type=int, default=5)
    parser.add_argument("--min-face-size", type=int, default=0)
    parser.add_argument("--out-path", type=str, default="./data")
    parser.add_argument("file")
    args = parser.parse_args()

    start_time = time()

    # Models found at: https://github.com/D2KLab/Face-Celebrity-Recognition
    facenet = tensorflow.keras.models.load_model("models/facenet_keras.h5")
    facenet.load_weights("models/facenet_keras_weights.h5")

    # Comment out 1, same wrapped api!
    # detector = MTCNNDetector()
    detector = RetinaFaceDetector(min_face_size=args.min_face_size)

    # Tracker - SORT. Has nothing to do with sorting
    multi_tracker = Sort(
        max_age=args.max_trajectory_age,
        min_hits=args.min_trajectory,
        iou_threshold=args.iou_threshold,
    )

    # Setup options and run extraction process
    options = Options(
        n_shards=args.n_shards,
        shard_i=args.shard_i,
        save_every=args.save_every,
        out_path=args.out_path.rstrip("/"),
        max_trajectory_age=args.max_trajectory_age,
        min_trajectory=args.min_trajectory,
    )
    process_video(args.file, options)

    minutes, seconds = divmod(time() - start_time, 60)
    print(f"Completed in {int(minutes)} minutes, {int(seconds)} seconds.")
