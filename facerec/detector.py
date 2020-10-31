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
    def __init__(self, min_face_size=0):
        """Face detector based on RetinaFace.

        Args:
            min_face_size (int): minimum size of a face to be considered a match.
                Compared against min(width, height) of the face bounding box.
        """
        self.min_size = min_face_size
        self.model = RetinaFace(quality="normal")

    def detect(self, img: np.array):
        assert len(img.shape) == 3 and img.shape[2] == 3
        bbs = self.model.predict(img, threshold=0.95)

        return [{
            "box": [b["x1"], b["y1"], b["x2"], b["y2"]],
            "keypoints": {
                "left_eye": (int(b["left_eye"][0]), int(b["left_eye"][1])),
                "right_eye": (int(b["right_eye"][0]), int(b["right_eye"][1])),
                "nose": (int(b["nose"][0]), int(b["nose"][1])),
                "mouth_left": (int(b["left_lip"][0]), int(b["left_lip"][1])),
                "mouth_right": (int(b["right_lip"][0]), int(b["right_lip"][1])),
            }
        } for b in bbs if min(b["x2"] - b["x1"], b["y2"] - b["y1"]) >= self.min_size]
