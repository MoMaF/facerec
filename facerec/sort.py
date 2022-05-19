"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    Original:
    github.com/abewley/sort/blob/7fc1ce2855ca0ea506b82a7f92ef8c0cf875e8d9/sort.py

    This is a slightly modified version of SORT, with the following changes:
    - Ability to stop a tracker without it reaching its max age
    - Require some number of real detection to start with, in order to continue tracks
    - Ability to deal with occlusions
    - Added some comments and fixed other oddities
"""
from __future__ import print_function

import sys
import os
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

debug = False

def linear_assignment(utility_matrix):
    """Solves the linear assignment problem.

    Returns: 2D matrix where the rows are the selected assignment indices (x, y).
    """
    x, y = linear_sum_assignment(utility_matrix, maximize=True)
    return np.array(list(zip(x, y))).astype(np.int32)


def iou_batch(bb_test, bb_gt):
    """Computes IUO between two bboxes in the form [x1, y1, x2, y2]"""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)

    intersection = w * h
    area_1 = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_2 = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_1 + area_2 - intersection

    return intersection / union


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    assert x.shape == (8, 1)  # Shape of kf internal state
    x = x[:,0]
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0])


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, first_frame):
        """Initialises a tracker using initial bounding box.
        """
        self.first_frame = first_frame

        # Init Kalman filter. State shape is:
        # x = [x_center, y_center, scale, ratio, *derivatives-of-those-4...]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        # give high uncertainty to the unobservable initial velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = [(self.get_state(), True)]
        self.time_since_update = 0
        self.hits = 1
        self.initial_hits = 1

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        # Overwrite the prediction update with the posterior estimate instead
        self.kf.update(convert_bbox_to_z(bbox))
        self.history[-1] = (self.get_state(), True)

        self.time_since_update = 0
        self.hits += 1
        if len(self.history) == self.hits:
            self.initial_hits += 1

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.

        Note: always call predict before an update.
        """
        # Area (x[6]) and ratio (x[7]) can't be below zero. Forcing positive.
        if (self.kf.x[6] + self.kf.x[2]) < 1e-3:
            self.kf.x[6] *= 0.0
        if (self.kf.x[7] + self.kf.x[3]) < 1e-3:
            self.kf.x[7] *= 0.0

        self.time_since_update += 1

        # Note: calling predict() also updates the prior estimate for x (self.kf.x)
        self.kf.predict()
        x = self.get_state()
        self.history.append((x, False))  # state + boolean saying if it's a posterior.
        return x

    def get_state(self):
        """
        Returns the current bounding box estimate in coords [x1, y1, x2, y2]
        """
        return convert_x_to_bbox(self.kf.x)

    def __len__(self):
        return len(self.history)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns (tuple):
        matches (indices d, t), unmatched_detections, unmatched_trackers
    """
    if len(trackers) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(trackers)),
        )

    # Assign optimal detection + tracker pairs
    iou_matrix = iou_batch(detections, trackers)
    iou_matrix[iou_matrix < iou_threshold] = -1.0
    matched_indices = linear_assignment(iou_matrix)

    # Filter matches to only contain those that pass the threshold
    matches = [m for m in matched_indices if iou_matrix[m[0], m[1]] >= iou_threshold]
    matches = np.array(matches).astype(matched_indices.dtype).reshape((-1, 2))

    # The rest are marked as unmatched
    unmatched_detections = np.array(list(set(range(len(detections))) - set(matches[:, 0])))
    unmatched_trackers = np.array(list(set(range(len(trackers))) - set(matches[:, 1])))

    return matches, unmatched_detections, unmatched_trackers


class Sort(object):
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.5):
        """
        Sets key parameters for SORT.

        Args:
            max_age: max age (in frames) to follow a trajectory with no real observations
                for a while.
            min_hits: minimum real observations needed in the beginning for a track to
                become valid.
            iou_threshold: minimum IoU for rectangles to be considered overlapping.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

        # These are bookeeping variables to tie trackers to detections/frames
        self.detection_count = 0
        self.tracker_id_map = {}  # tracker id -> list of detection ids
        self.detection_id_map = {}  # detection id -> tracker (object)
        self.frame_map = {}  # detection id -> frame index

    def update(self, detections: np.array, frame: int):
        """Update all trackers with detections from a frame.

        Note: call with all frames, even ones with no detections.

        Args:
            detections: a numpy array of detections in the format [x1, y1, x2, y2, score]
            frame: frame index, to keep track of updates

        Returns:
            An np.array with globally unique indices of trackers that each detection
                was assigned.
        """
        self.frame_count += 1
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            print("WARNING: Removed tracker with NaN predictions.")
            self.trackers[t].had_nan_preds = True
            self.trackers.pop(t)

        assert detections.shape[1] == 5
        matched, unmatched_dets, _ = associate_detections_to_trackers(
            detections, trks, self.iou_threshold
        )

        # Assign new globally unique indices to each detection
        detections_idx = self.detection_count + np.arange(len(detections))
        self.detection_count += len(detections)
        # Remember the frame index of each detection
        for detection_id in detections_idx:
            self.frame_map[detection_id] = frame

        # Update matched trackers with assigned detections
        for det_index, trk_index in matched:
            trk: KalmanBoxTracker = self.trackers[trk_index]
            trk.update(detections[det_index])
            prev = self.tracker_id_map[trk.id][-1]
            self.tracker_id_map[trk.id].append(detections_idx[det_index])
            self.detection_id_map[detections_idx[det_index]] = trk
            if debug:
                print('updated', detections_idx[det_index], prev, det_index, trk_index)

        # Unfollow trackers that have "expired" (note: still exist in detection_id_map)
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            expired = trk.time_since_update > self.max_age and len(trk) >= self.min_hits
            not_started = len(trk) <= self.min_hits and trk.initial_hits < len(trk)
            if expired or not_started:
                self.trackers.pop(i)

        # Create and initialise new trackers for unmatched detections
        for det_index in unmatched_dets:
            trk = KalmanBoxTracker(detections[det_index], frame)
            trk.had_nan_preds = False
            self.trackers.append(trk)
            self.tracker_id_map[trk.id] = [detections_idx[det_index]]
            self.detection_id_map[detections_idx[det_index]] = trk
            if debug:
                print('added', detections_idx[det_index], det_index, frame)

        # Return the new globally unique indices for each detection
        return detections_idx

    def has_valid_tracker(self, detection_id):
        """Indicates whether a tracker associated with an id is valid.

        Note: the tracker could be either active or inactive/expired, and still
        be valid.
        """
        trk = self.detection_id_map.get(detection_id)
        assert trk is not None, "Tried to access non-existent tracker <"+str(detection_id)+">"

        # TODO: more criteria?
        start_ok = trk.initial_hits >= self.min_hits and not trk.had_nan_preds
        return start_ok

    def has_valid_tracker_safe(self, detection_id):
        """Indicates whether a tracker associated with an id is valid.

        Note: the tracker could be either active or inactive/expired, and still
        be valid.
        """
        if debug:
            print('detection_id_map', list(self.detection_id_map.keys()))
        trk = self.detection_id_map.get(detection_id)
        return trk is not None

    def get_detection_bbox(self, detection_id):
        """Get the filtered (by Kalman) bbox for a detection.
        """
        trk = self.detection_id_map.get(detection_id)
        assert trk is not None, "Tried to access non-existent tracker!"
        detection_frame = self.frame_map.get(detection_id)
        i = detection_frame - trk.first_frame

        assert i >= 0 and i < len(trk), "Faulty frame index!"
        bbox, _ = trk.history[i]
        return bbox

    def kill_trackers(self):
        """Trigger to kill all trackers at once.
        """
        self.trackers = []

    def pop_expired(self, expiry_age: int, current_frame: Optional[int] = None):
        """Allow SORT to remove trackers stored internally.

        Args:
            expiry_age: (int) A tracker can be removed if it had no detections for this
                many frames recently.
            current_frame: Current frame (int), used to compute the age of the tracker.
                Set to None to force expire of all trackers.
        """
        if current_frame is None:
            current_frame = sys.maxsize

        expired_trackers = []
        for trk_id in list(self.tracker_id_map.keys()):
            detection_ids = self.tracker_id_map[trk_id]
            if debug:
                print('detection_ids', trk_id, detection_ids)
            trk = self.detection_id_map[detection_ids[0]]
            trk_age = current_frame - (trk.first_frame + len(trk) - trk.time_since_update - 1)
            assert trk_age >= 0, "Age less than zero?"
            if trk_age >= expiry_age:
                # Clean up internal structures
                del self.tracker_id_map[trk_id]
                for det_id in detection_ids:
                    del self.detection_id_map[det_id]
                    if debug:
                        print('deleted', det_id, trk_id, trk_age, expiry_age, current_frame, trk.first_frame, len(trk), trk.time_since_update)
                    del self.frame_map[det_id]

                # Add valid trackers to the list we'll return
                if trk.initial_hits >= self.min_hits:
                    # Remove predicted stuff at the end that weren't observations
                    trk.history = trk.history[:len(trk) - trk.time_since_update]
                    expired_trackers.append(trk)
            elif debug:
                for det_id in detection_ids:
                    print('not deleted', det_id, trk_id, trk_age, expiry_age, current_frame, trk.first_frame, len(trk), trk.time_since_update)
                    
        return expired_trackers
