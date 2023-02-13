#! /usr/bin/env python3

import sys
import cv2
import numpy as np
from pymediainfo import MediaInfo

cap = cv2.VideoCapture(sys.argv[1])

storage_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
storage_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

try:
    print('--------- Using MediaInfo -----------')
    video_info_json_str = MediaInfo.parse(file, output="JSON")
    tracks = json.loads(video_info_json_str)["media"]["track"]
    video_info = next(track for track in tracks if track["@type"].lower() == "video")
    dar_string = video_info["DisplayAspectRatio_String"]

    if ":" in dar_string:
        num, den = [float(s) for s in dar_string.split(":")]
        dar = num / den
    else:
        dar = float(dar_string)
except:
    print('------ "VLC" approach as fallback -------')
    sar = storage_w / storage_h
    numerator = cap.get(cv2.CAP_PROP_SAR_NUM) or 1.0
    denominator = cap.get(cv2.CAP_PROP_SAR_DEN) or 1.0
    par = numerator / denominator
    dar = sar * par

# Compute the required resolution with DAR aspect ratio.
dar_h = storage_h
dar_w = round(dar_h * dar)
print(f"Display aspect ratio: {dar:.2f} (resolution: {dar_w}×{dar_h})")

# Read image in storage resolution
_, frame = cap.read()

# If needed, scale it to DAR!
if dar_w != storage_w:
    frame = cv2.resize(frame, (dar_w, dar_h))
    assert frame.shape[0] == dar_h and frame.shape[1] == dar_w

frame_img: np.array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# CORRECT! This is even better than the last solution!
# Here, `frame_img` will have the shape (dar_h, dar_w, 3)
# objects = my_object_detector.pad_and_predict(frame_img)

cap.release()

# 0100447.m4v

# extract.py:
# fps=25.0 video_w=710 video_h=574 sar=1.2369337979094077 num=16.0 den=15.0 par=1.0666666666666667 dar=1.3193960511033682 d_width=757

# this:
# Display aspect ratio: 1.32 (resolution: 757×574)
