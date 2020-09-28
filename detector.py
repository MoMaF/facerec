"""Wrappers for 2 common face detectors: MTCNN and RetinaFace.

Standard return format: {
    "box": [x1, y1, x2, y2],
    "keypoints": {
        left_eye: (x, y),
        right_eye: (x, y),
        nose: (x, y),
        mouth_left: (x, y),
        mouth_right: (x, y),
    }
}
"""
import mtcnn
from retinaface import RetinaFace
import numpy as np

class MTCNNDetector:
    def __init__(self):
        self.model = mtcnn.MTCNN(min_face_size=20)

    def detect(self, img: np.array):
        assert len(img.shape) == 3 and img.shape[2] == 3
        bbs = self.model.detect_faces(img)

        # Convert from x1, y1, w, h --> x1, y1, x2, y2 for boxes
        for b in bbs:
            del b["confidence"]
            b["box"][2] += b["box"][0]
            b["box"][3] += b["box"][1]

        return bbs

class RetinaFaceDetector:
    def __init__(self):
        self.model = RetinaFace(quality="normal")

    def detect(self, img: np.array):
        assert len(img.shape) == 3 and img.shape[2] == 3
        bbs = self.model.predict(img, threshold=0.95)

        return [{
            "box": [b["x1"], b["y1"], b["x2"], b["y2"]],
            "keypoints": {
                "left_eye": b["left_eye"],
                "right_eye": b["right_eye"],
                "nose": b["nose"],
                "mouth_left": b["left_lip"],
                "mouth_right": b["right_lip"],
            }
        } for b in bbs]
