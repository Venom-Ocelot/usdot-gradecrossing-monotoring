"""
ZOI (Zone of Interest) utilities.

Handles feature detection, optical flow tracking, and ZOI polygon stabilization.
"""

import cv2
import numpy as np

from . import config


def detect_feature_points(gray: np.ndarray) -> np.ndarray:
    """Detect stable background feature points in a grayscale frame."""
    return cv2.goodFeaturesToTrack(gray, **config.FEATURE_PARAMS)


def stabilize_zoi(
    gray: np.ndarray,
    first_gray: np.ndarray,
    first_points: np.ndarray,
    zoi_points: np.ndarray,
) -> np.ndarray:
    """Warp zoi_points from the anchor frame to the current frame.

    Tracks first_points from the first frame into the current frame using
    Lucas-Kanade optical flow, filters outliers with RANSAC, and applies
    the resulting homography to the original ZOI polygon.

    Falls back to the original zoi_points if tracking fails.
    """
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        first_gray, gray, first_points, None, **config.LK_PARAMS
    )
    good_first = first_points[status == 1]
    good_curr  = curr_points[status == 1]

    if len(good_first) >= 4:
        H, _ = cv2.findHomography(good_first, good_curr, cv2.RANSAC, 3.0)
        if H is not None:
            zoi_reshaped = zoi_points.astype(np.float32).reshape(-1, 1, 2)
            return cv2.perspectiveTransform(zoi_reshaped, H).reshape(-1, 2).astype(np.int32)

    return zoi_points.copy()


def build_zoi_mask(frame_shape: tuple, zoi_points: np.ndarray) -> np.ndarray:
    """Return a filled binary mask for the ZOI polygon."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [zoi_points], 255)
    return mask
