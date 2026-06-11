"""
Pipeline configuration — default values for all tunable parameters.

Import this module and override individual values before passing to runner.run_pipeline().
"""

import cv2

# ── Model weights ──────────────────────────────────────────────────────────
MODEL_PATH = "weights/best.pt"

# ── Optical flow (Lucas-Kanade) ─────────────────────────────────────────────
FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=30,
    blockSize=7,
)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),  # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
)

# ── MOG2 background subtractor ──────────────────────────────────────────────
MOG2_HISTORY       = 300   # frames used to learn the background
MOG2_VAR_THRESHOLD = 50    # sensitivity — higher = less sensitive
MOG2_DETECT_SHADOWS = True

# ── Safety thresholds ───────────────────────────────────────────────────────
AREA_THRESHOLD = 5000  # minimum debris pixel area to start the debris timer
# vehicle_threshold and time_threshold are set per-video in data/manifests/videos.yaml

# ── Detection stride & confidence ─────────────────────────────────────────
YOLO_STRIDE = 5     # run YOLO every N frames
YOLO_CONF   = 0.25  # minimum detection confidence (0–1)

# ── Snapshot capture ────────────────────────────────────────────────────────
LATE_SNAPSHOT_TIME = 28.0  # seconds into video for the late-run snapshot
