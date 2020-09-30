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

import face_utils
import utils
from detector import MTCNNDetector, RetinaFaceDetector

CROP_MARGIN = 0
FACE_IMAGE_SIZE = 160  # save face crops in this image resolution! (required!)

Options = namedtuple(
    "Options",
    ["out_path", "n_shards", "shard_i", "save_every", "iou_threshold", "min_trajectory"],
)

def iou(boxA, boxB):
    """Intersection Over Union between the areas of 2 rectangles/boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

def save_trajectory(file, trajectory, min_trajectory_len):
    if len(trajectory["faces"]) < min_trajectory_len:
        return False
    # Write out .jsonl
    bbs = [face["box"] for face in trajectory["faces"]]
    out_obj = {"start": trajectory["start"], "len": len(trajectory["faces"]), "bbs": bbs}
    json.dump(out_obj, file, indent=None, separators=(",", ":"))
    file.write("\n")
    return True

def process_frame(frame_data, trajectories, features_file, images_dir, min_trajectory_len):
    """Save faces + features from a frame, and creating face embeddings.
    """
    # Filter to faces with a valid trajectory (len > MIN)
    frame_data["faces"] = [
        f for f in frame_data["faces"]
        if len(trajectories[f["trajectory"]]["faces"]) >= min_trajectory_len
    ]

    img = Image.fromarray(frame_data["img_np"])
    for face in frame_data["faces"]:
        # Crop onto face only
        cropped = img.crop(tuple(face["box"]))
        resized = cropped.resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)

        # Get face embedding vector via facenet model
        scaledx = np.array(resized)
        scaledx = scaledx.reshape(-1, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE, 3)
        embedding = utils.get_embedding(facenet, scaledx[0])

        # Save face image and features
        box_label = frame_data["label"] + "_{}_{}_{}_{}".format(*face["box"])
        resized.save(f"{images_dir}/kept-{box_label}.jpeg")
        json.dump({
            "frame": frame_data["index"],
            "label": box_label,
            "embedding": embedding.tolist(),
            "box": face["box"],
            "keypoints": face["keypoints"],
        }, features_file, indent=None, separators=(",", ":"))
        features_file.write("\n")

    return len(frame_data["faces"])

def process_video(file, opt: Options):
    """Process entire video and extract face boxes.
    """
    assert opt.shard_i < opt.n_shards and opt.shard_i >= 0, "Bad shard index."

    cap = cv2.VideoCapture(file)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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

    # Trajectories = same face across multiple frames
    max_ti = -1
    trajectories = {}

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

        # Are faces in a known trajectory?
        # Active trajectory: the last face is from the previous frame processed
        active_trajectories = {
            id: traj for id, traj in trajectories.items()
            if traj["start"] + len(traj["faces"]) == f
        }
        for face in faces:
            found_ti = None
            best_iou = -1
            for ti, traj in active_trajectories.items():
                iou_value = iou(face["box"], traj["faces"][-1]["box"])
                if iou_value > best_iou:
                    found_ti = ti
                    best_iou = iou_value

            if best_iou > opt.iou_threshold:
                trajectories[found_ti]["faces"].append(face)
                face["trajectory"] = found_ti
            else:
                # create new trajectory
                max_ti += 1
                trajectories[max_ti] = {"faces": [face], "start": f}
                face["trajectory"] = max_ti

        if len(set([face["trajectory"] for face in faces])) != len(faces):
            print("WARNING: Trajectory mismatch")

        # Clean up expired trajectories
        for ti in list(trajectories.keys()):
            traj = trajectories[ti]
            if traj["start"] + len(traj["faces"]) < f - opt.min_trajectory:
                saved_traj_count += int(save_trajectory(trajectories_file, traj, opt.min_trajectory))
                del trajectories[ti]

        # Extract good face boxes from middle frame, save those
        if len(buf) == opt.min_trajectory:
            frame_data = buf.pop(0)
            if frame_data["index"] % opt.save_every == 0:
                n_saved_faces = process_frame(
                    buf.pop(0), trajectories, features_file, images_dir, opt.min_trajectory
                )
                saved_boxes_count += n_saved_faces
                saved_frames_count += int(n_saved_faces > 0)

    # Save remaining frames and trajectories
    for frame_data in buf:
        if frame_data["index"] % opt.save_every == 0:
            n_saved_faces = process_frame(
                frame_data, trajectories, features_file, images_dir, opt.min_trajectory
            )
            saved_boxes_count += n_saved_faces
            saved_frames_count += int(n_saved_faces > 0)
    for ti, traj in trajectories.items():
        saved_traj_count += int(save_trajectory(trajectories_file, traj, opt.min_trajectory))

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
    parser.add_argument("--out-path", type=str, default=".")
    parser.add_argument("--min-trajectory", type=int, default=5)
    parser.add_argument("file")
    args = parser.parse_args()

    start_time = time()

    facenet = tensorflow.keras.models.load_model("facenet_keras.h5")
    facenet.load_weights("facenet_keras_weights.h5")

    # Comment out 1, same wrapped api!
    # detector = MTCNNDetector()
    detector = RetinaFaceDetector()

    _, ext = os.path.splitext(os.path.basename(args.file))
    if ext in [".mpeg", ".mpg", ".mp4", ".avi", ".wmv"]:
        options = Options(
            n_shards=args.n_shards,
            shard_i=args.shard_i,
            save_every=args.save_every,
            iou_threshold=args.iou_threshold,
            min_trajectory=args.min_trajectory,
            out_path=args.out_path,
        )
        process_video(args.file, options)

        minutes, seconds = divmod(time() - start_time, 60)
        print(f"Completed in {int(minutes)} minutes, {int(seconds)} seconds.")
