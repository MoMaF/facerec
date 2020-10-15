"""Merge trajectories from different shards created by extract.py.
"""
import argparse
import os
import json
from extract import iou, save_trajectory

def save_trajectory(file, trajectory):
    # Write out .jsonl
    json.dump(trajectory, file, indent=None, separators=(",", ":"))
    file.write("\n")

def merge(trajectories_dir, out_dir, iou_threshold):
    """Merge trajectories that cross file boundaries in terms of frames.
    """
    if not os.path.exists(trajectories_dir):
        return
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

    print(f"Processing {len(traj_files)} trajectory files.")

    out_file = open(os.path.join(out_dir, "trajectories.jsonl"), "w")
    trajectories = []

    n_read = 0
    n_saved = 0
    n_merges = 0

    for file in traj_files:
        with open(file["path"], "r") as f:
            new_trajectories = [json.loads(line) for line in f]
            n_read += len(new_trajectories)
        mergables = [t for t in new_trajectories if t["start"] == file["s"]]
        others = [t for t in new_trajectories if t["start"] != file["s"]]

        expired = [t for t in trajectories if (t["start"] + t["len"]) < file["s"]]
        trajectories = [t for t in trajectories if (t["start"] + t["len"]) == file["s"]]

        # Save trajectories that can't be merged anymore, to disk
        for trajectory in expired:
            n_saved += 1
            save_trajectory(out_file, trajectory)

        # Check if some of the new trajectories can merge into an old one
        for t1 in mergables:
            best_iou = iou_threshold
            best_t = None
            for t2 in trajectories:
                iou_value = iou(t2["bbs"][-1], t1["bbs"][0])
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_t = t2

            # A merge was found!
            if best_t is not None:
                n_merges += 1
                best_t["bbs"] += t1["bbs"]
                best_t["len"] = len(best_t["bbs"])
            else:
                trajectories.append(t1)

        trajectories += others

    for trajectory in trajectories:
        n_saved += 1
        save_trajectory(out_file, trajectory)

    out_file.close()
    print(f"Done! Read {n_read} trajectories and saved {n_saved}. (Total merges: {n_merges})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--path", type=str, default=".")
    args = parser.parse_args()

    data_dir = args.path
    trajectories_dir = os.path.join(data_dir, "trajectories")
    assert os.path.exists(trajectories_dir), f"Didn't find: {trajectories_dir}"

    merge(trajectories_dir, data_dir, args.iou_threshold)
