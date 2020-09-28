import os
import sys
import argparse
from collections import namedtuple

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
    ["out_path", "n_shards", "shard_i", "save_every", "margin", "iou_threshold"],
)

def process_image(file):
    label,_ = os.path.splitext(os.path.basename(file))
    img = Image.open(file).convert('RGB')
    return process_frame(np.asarray(img), label, 0)

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

def process_frame_buffer(buf, iou_threshold=0.5):
    """Process small frame buffer to check that face detections have time consistency
    ie. they appear roughly in the same place when compared to the middle frame.

    Args:
        buf: list of dicts with frame data.
        iou_threshold: intersection over union - threshold for a match.
    """
    # Extract the middle frame against which we'll compare the other frames
    midx  = len(buf) // 2
    mid_frame = buf[midx].copy()
    other_frames = buf[:midx] + buf[(midx + 1):]
    ok_mid_boxes = []

    # Loop to check that every other frame in buffer has a matching box, when
    # compared to the target middle frame. If yes: keep it!
    for mid_box_info in mid_frame["boxes"]:
        keep = True
        for other_frame in other_frames:
            found = False
            for other_box_info in other_frame["boxes"]:
                iu = iou(mid_box_info["box"], other_box_info["box"])
                if iu > iou_threshold:
                    found = True
                    break
            if not found:
                keep = False
                break
        if keep:
            ok_mid_boxes.append(mid_box_info)

    mid_frame["boxes"] = ok_mid_boxes
    return mid_frame

def process_frame(npimg, label, index):
    """Process a single video frame and find appearances of faces in it.
    """
    img_shape = npimg.shape

    img  = Image.fromarray(npimg)
    imgb = img.copy()
    draw = ImageDraw.Draw(imgb)

    # Boxed = same image with white boxes around faces (drawn below...)
    frame_data = {"label": label, "index": index, "image": img, "boxed": imgb}
    boxes_metadata = []

    faces = detector.detect(npimg)
    for i, face in enumerate(faces):
        draw.rectangle(face["box"], fill=None, outline=None)

        # Rectangle: x1, y1, x2, y2 + margin for output crop images
        rect = np.array(face["box"]).astype(np.int32)
        rect[:2] = np.maximum(rect[:2] - CROP_MARGIN, [0, 0])
        rect[2:] = np.minimum(rect[2:] + CROP_MARGIN, [img_shape[1], img_shape[0]])

        # Crop onto face only
        cropped = img.crop(tuple(rect))
        resized = cropped.resize((FACE_IMAGE_SIZE, FACE_IMAGE_SIZE), resample=Image.BILINEAR)
        scaled = np.array(resized.convert('L'))

        # Get face embedding vector via facenet model
        scaledx = np.array(resized)
        scaledx = scaledx.reshape(-1, FACE_IMAGE_SIZE, FACE_IMAGE_SIZE, 3)
        embedding = utils.get_embedding(facenet, scaledx[0])

        # Store meta + image data about the face rectangle
        box_info = {
            "box": rect,
            "scaled": scaled,
            "label": label + '_{}_{}_{}_{}'.format(*rect),
            "features": embedding,
        }
        boxes_metadata.append(box_info)

    frame_data["boxes"] = boxes_metadata
    return frame_data

def save_face_features(frame_data, out_file):
    for box_info in frame_data['boxes']:
        out_file.write(f"{frame_data['label']},")
        out_file.write(f"{box_info['label']},")
        out_file.write(",".join(map(str, box_info["features"])))
        out_file.write("\n")

def save_frame_images(frame_data, save_face_boxes, save_boxed_frame, out_dir):
    if save_face_boxes:
        for box_info in frame_data['boxes']:
            path = f"{out_dir}/{box_info['label']}.jpeg"
            Image.fromarray(box_info['scaled']).save(path)
    if save_boxed_frame:
        path = f"{out_dir}/{frame_data['label']}_boxed.jpeg"
        frame_data['boxed'].save(path)

def process_video(file, opt: Options):
    """Process entire video and extract face boxes.
    """
    assert opt.shard_i < opt.n_shards and opt.shard_i >= 0, "Bad shard index."

    cap = cv2.VideoCapture(file)
    n_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    shard_len = (n_total_frames + opt.n_shards - 1) // opt.n_shards
    beg = shard_len * opt.shard_i
    end = beg + shard_len # not inclusive

    label, _ = os.path.splitext(os.path.basename(file))
    features_dir = f"{opt.out_path}/{label}-data/features"
    images_dir = f"{opt.out_path}/{label}-data/images"
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    features_path = f"{features_dir}/features_{label}_{beg}-{end}.txt"
    features_file = open(features_path, mode="w")

    buf = []
    max_buffer_size = 2 * opt.margin + 1

    print(f"Movie file: {os.path.basename(file)}")
    print(f"Total length: {(n_total_frames / fps / 3600):.1f}h ({fps} fps)")
    print(f"Shard {(opt.shard_i + 1)} / {opt.n_shards}, len: {shard_len} frames")
    print(f"Processing frames: {beg} - {end} (max: {n_total_frames})")

    beg_with_margin = max(beg - opt.margin, 0)
    end_with_margin = min(end + opt.margin, n_total_frames)
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, beg_with_margin), \
        f"Couldn't set start frame to: {beg_with_margin}"

    saved_frames_count = 0
    saved_boxes_count = 0
    for f in range(beg_with_margin, end_with_margin):
        ret, frame = cap.read()

        if not ret:
            break

        frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_label = label + ':' + str(f).zfill(6)
        frame_data = process_frame(frame_img, frame_label, f)

        # Store frame data in a small buffer that we'll process all at once
        buf.append(frame_data)

        # Extract good face boxes from middle frame, save those
        if len(buf) == max_buffer_size:
            mid_frame_index = f - opt.margin
            if mid_frame_index % opt.save_every == 0:
                middle_frame = process_frame_buffer(buf, opt.iou_threshold)
                for box_info in middle_frame["boxes"]:
                    box_info["label"] = "kept-" + box_info["label"]
                if len(middle_frame["boxes"]) > 0:
                    saved_frames_count += 1
                    saved_boxes_count += len(middle_frame["boxes"])
                    save_frame_images(middle_frame, True, True, images_dir)
                    save_face_features(middle_frame, features_file)
            buf.pop(0)

    features_file.close()
    cap.release()
    print(f"Saved {saved_boxes_count} boxes from {saved_frames_count} different frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--n-shards", type=int, default=256)
    parser.add_argument("--shard-i", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--margin", type=int, default=2)
    parser.add_argument("--iou-threshold", type=float, default=0.75)
    parser.add_argument("--out-path", type=str, default=".")
    parser.add_argument("file")
    args = parser.parse_args()

    facenet = tensorflow.keras.models.load_model('facenet_keras.h5')
    facenet.load_weights('facenet_keras_weights.h5')

    # Comment out 1, same wrapped api!
    # detector = MTCNNDetector()
    detector = RetinaFaceDetector()

    _, ext = os.path.splitext(os.path.basename(args.file))
    if ext in ['.mpeg', '.mpg', '.mp4', '.avi', '.wmv']:
        options = Options(
            n_shards=args.n_shards,
            shard_i=args.shard_i,
            save_every=args.save_every,
            margin=args.margin,
            iou_threshold=args.iou_threshold,
            out_path=args.out_path,
        )
        process_video(args.file, options)
