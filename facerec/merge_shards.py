#! /usr/bin/env python3

"""Merge data produced in different shards of extraction.

Namely:
    1. Trajectories
    2. Scene cuts

Output: single merged files trajectories.jsonl and scene_changes.json
"""
from typing import Set
import argparse
import os
import json
import glob

from utils.utils import load_images_map

def is_trajectory_valid(trajectory, images_map):
    """Check that a trajectory has associated images.
    """
    # TODO: add min count instead?
    for frame_index, bbs in enumerate(trajectory["bbs"], start=trajectory["start"]):
        # print(frame_index, bbs, frame_index in images_map, tuple(bbs) in images_map[frame_index])
        if frame_index in images_map and tuple(bbs) in images_map[frame_index]:
            return True
    # print('not valid', trajectory)
    return False

def passes_min_size(trajectory, min_face_size):
    """Check that faces have a certain minimum (pixel) size. This is useful if
    we want to have reliable embedddings of images in a trajectory.
    """
    for bbs in trajectory["bbs"]:
        # Bounding boxes (bbs) are: x1, y1, x2, y2
        w, h = (bbs[2] - bbs[0]), (bbs[3] - bbs[1])
        # print(w, h, min_face_size)
        if min(w, h) >= min_face_size:
            return True
    # print('not enough size', trajectory)
    return False

def passes_min_size_old(trajectory, min_face_size):
    """Check that faces have a certain minimum (pixel) size. This is useful if
    we want to have reliable embedddings of images in a trajectory.
    """
    for bbs in trajectory["bbs"]:
        # Bounding boxes (bbs) are: x1, y1, x2, y2
        w, h = (bbs[2] - bbs[0]), (bbs[3] - bbs[1])
        # print(w, h, min_face_size)
        if min(w, h) < min_face_size:
            # print('not enough size', trajectory)
            return False
    return True

def save_trajectories(file, trajectories, images_map, min_face_size, traj_count, movie_id):
    """Save trajectories, and filter out trajectories that had no corresponding
    images for any bounding boxes.
    """
    # Write out .jsonl
    n_saved = 0
    for traj in trajectories:
        # print('x', traj)
        if is_trajectory_valid(traj, images_map) and passes_min_size(traj, min_face_size):
            traj["index"] = traj_count
            traj["movie_id"] = movie_id
            json.dump(traj, file, indent=None, separators=(",", ":"))
            file.write("\n")
            traj_count += 1
            n_saved += 1
    n_removed = len(trajectories) - n_saved
    return n_saved, n_removed

def save_scene_changes(file_path, scene_cuts: Set[int], movie_id: int):
    scene_cuts_list = sorted(scene_cuts)
    with open(file_path, "w") as file:
        obj = {"frame_indices": scene_cuts_list, "movie_id": movie_id}
        json.dump(obj, file, indent=None, separators=(",", ":"))
        file.write("\n")

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

def load_trajectory(trajectory_file: str, scene_cuts: Set[int], iou_threshold: float):
    """Load a single trajectory file (from 1 shard).

    Note: also merges possible trajectories that weren't caught by the face trackers
    earlier, while making sure that no trajectories pass scene_cuts.
    """
    with open(trajectory_file, "r") as f:
        trajectories = sorted([json.loads(line) for line in f], key=lambda t: t["start"])

    # for t in trajectories:
    #     print('a', trajectory_file, t)
        
    merged_trajectories = []
    merged_indices = set()

    # Loop to merge trajectories within the shard itself
    for i, t1 in enumerate(trajectories):
        if i in merged_indices:
            continue
        found_merge = True
        while found_merge:
            end = t1["start"] + t1["len"]
            best_iou = iou_threshold
            best_j = None
            for j, t2 in enumerate(trajectories[i + 1:], start=i + 1):
                if t2["start"] != end or j in merged_indices or end in scene_cuts:
                    continue
                iou_value = iou(t1["bbs"][-1], t2["bbs"][0])
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_j = j
            found_merge = (best_j is not None)
            if found_merge:
                t1["bbs"] = t1["bbs"] + trajectories[best_j]["bbs"]
                t1["detected"] = t1["detected"] + trajectories[best_j]["detected"]
                t1["len"] = len(t1["bbs"])
                merged_indices.add(best_j)
        merged_trajectories.append(t1)

    # for t in merged_trajectories:
    #     print('b', trajectory_file, t)
        
    # Return final trajectories + number of merges made
    n_merges = len(trajectories) - len(merged_trajectories)
    return merged_trajectories, n_merges

def merge(
    data_dir: str, movie_id: int, iou_threshold: float, overlap: int, min_face_size: int
):
    """Merge trajectories that cross file boundaries in terms of frames.
    """
    trajectories_dir = os.path.join(data_dir, "trajectories")
    scene_changes_dir = os.path.join(data_dir, "scene_changes")
    features_dir = os.path.join(data_dir, "features")
    images_dir = os.path.join(data_dir, "images")
    assert os.path.exists(trajectories_dir), f"Didn't find: {trajectories_dir}"
    assert os.path.exists(scene_changes_dir), f"Didn't find: {scene_changes_dir}"
    assert os.path.exists(features_dir), f"Didn't find: {features_dir}"
    assert os.path.exists(images_dir), f"Didn't find: {images_dir}"

    # Check what trajectory files we have (one for each shard)
    _, _, filenames = next(os.walk(trajectories_dir))
    traj_files = []
    for file in filenames:
        # file is like: trajectories_987654_1000-2000.jsonl
        name, ext = os.path.splitext(file)
        parts = name.split("_")
        if parts[0] != "trajectories":
            continue
        start, end = [int(f) for f in parts[2].split("-")]
        traj_files.append({"s": start, "e": end, "path": os.path.join(trajectories_dir, file)})
    traj_files = sorted(traj_files, key=lambda d: d["s"])

    # Check that we have corresponding scene cut files (one for each shard)
    scene_cuts = set()
    for t_file in traj_files:
        start, end = t_file["s"], t_file["e"]
        # Scene change files have the same format as trajectory files
        filename = f"scene_changes_{movie_id}_{start}-{end}.json"
        scene_change_file = os.path.join(scene_changes_dir, filename)
        if os.path.exists(scene_change_file):
            with open(scene_change_file, "r") as f:
                shard_scene_cuts = json.load(f)["frame_indices"]
                scene_cuts |= set(shard_scene_cuts)

    # Merge feature files into one
    _, _, filenames = next(os.walk(features_dir))
    feature_files = []
    for file in filenames:
        # file is like: features_987654_1000-2000.jsonl
        name, ext = os.path.splitext(file)
        parts = name.split("_")
        if parts[0] != "features":
            continue
        start, _ = [int(f) for f in parts[2].split("-")]
        feature_files.append({"s": start, "path": os.path.join(features_dir, file)})
    feature_files = sorted(feature_files, key=lambda f: f["s"])

    with open(os.path.join(data_dir, "features.jsonl"), "w") as write_file:
        for file_obj in feature_files:
            with open(file_obj["path"], "r") as read_file:
                write_file.write(read_file.read())

    print(f"Processing {len(traj_files)} trajectory files.")
    print(f"Read a total {len(scene_cuts)} scene changes.")

    # Load image lookup map that allows to check if a frame + bbs combo has an image
    image_map = load_images_map(images_dir, features_dir)
    print(f"Read {len(image_map)} images.")

    out_file = open(os.path.join(data_dir, "trajectories.jsonl"), "w")
    trajectories = []

    n_read = 0
    n_saved = 0
    n_merges = 0
    n_deleted = 0

    # Loop to merge trajectories across different shards
    for file in traj_files:
        new_trajectories, n_shard_merges = load_trajectory(file["path"], scene_cuts, iou_threshold)
        n_read += len(new_trajectories)
        n_merges += n_shard_merges

        mergables = [t for t in new_trajectories if t["start"] < file["s"] + overlap]
        others = [t for t in new_trajectories if t["start"] >= file["s"] + overlap]

        # for t in mergables:
        #     print('c', t)
        # for t in others:
        #     print('d', t)

        expired = [t for t in trajectories if (t["start"] + t["len"]) < file["s"]]
        trajectories = [t for t in trajectories if (t["start"] + t["len"]) >= file["s"]]

        # Save trajectories that can't be merged anymore, to disk
        ns, nr = save_trajectories(out_file, expired, image_map, min_face_size, n_saved, movie_id)
        n_saved += ns
        n_deleted += nr

        # Check if some of the new trajectories can merge into an old one
        for t1 in mergables:
            best_iou = iou_threshold
            best_t = None

            # We'll only attempt a merge if t1["start"] isn't at a scene cut
            if t1["start"] not in scene_cuts:
                for t2 in trajectories:
                    if t2["start"] >= t1["start"] or (t2["start"] + t2["len"]) <= t1["start"]:
                        continue
                    t2_bbs_i = t1["start"] - t2["start"]
                    assert t2_bbs_i >= 0, "Invalid index?"
                    iou_value = iou(t2["bbs"][t2_bbs_i], t1["bbs"][0])
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_t = t2

            # A merge was found!
            if best_t is not None:
                # print('e', t1)
                # print('ex', best_t)
                n_merges += 1
                assumed_len = t1["start"] + t1["len"] - best_t["start"]
                best_t["bbs"] = best_t["bbs"][:(t1["start"] - best_t["start"])] + t1["bbs"]
                best_t["detected"] = best_t["detected"][:(t1["start"] - best_t["start"])] + t1["detected"]
                best_t["len"] = len(best_t["bbs"])
                assert best_t["len"] == assumed_len, "Len???"
            else:
                others.append(t1)
                # print('f', t1)

        trajectories += others

    # Save remaining
    ns, nr = save_trajectories(out_file, trajectories, image_map, min_face_size, n_saved, movie_id)
    n_saved += ns
    n_deleted += nr
    out_file.close()

    # Save merged scene cuts
    scene_cuts_file = os.path.join(data_dir, "scene_changes.json")
    save_scene_changes(scene_cuts_file, scene_cuts, movie_id)

    print(f"Total merges: {n_merges}.")
    print(f"Total removed if they had no images or had too small faces: {n_deleted}.")
    print(f"Done! Read {n_read} trajectories and saved {n_saved}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IOU threshold when merging bounding boxes.")
    parser.add_argument("--overlap", type=int, default=5,
                        help="""Overlap to consider when merging across shards. Should
                        match the max-trajectory-age that was used when extracting.""")
    parser.add_argument("--min-face-size", type=int, default=50,
                        help="""If bigger than zero, will filter trajectories that
                        have faces where `min(w, h) < min-face-size`.""")
    parser.add_argument("--path", type=str, default=".")
    args = parser.parse_args()

    data_dirs = glob.glob(args.path)
    for data_dir in data_dirs:
        # data_dir will be a movie directory like ./data/123456-data
        data_dir = data_dir.rstrip("/")
        print(f"Merging shards in: {data_dir}")

        movie_id: int = int(os.path.basename(data_dir).split("-")[0])

        merge(data_dir, movie_id, args.iou_threshold, args.overlap, args.min_face_size)
        print()
