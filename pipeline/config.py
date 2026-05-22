"""
Pipeline configuration — default values for all tunable parameters.

Import this module and override individual values before passing to runner.run_pipeline().
"""

# ── Video / Model inputs ────────────────────────────────────────────────────
VIDEO_PATH  = "data/cache/crossing_001.mp4"
MODEL_PATH  = "weights/best.pt"
RUN_LABEL   = "run002_finetuned"

# ── Zone of Interest ────────────────────────────────────────────────────────
# Four corners of the ZOI polygon with pixel coordinates
ZOI_POINTS = [
    [100,  400],  # Top Left
    [500,  350],  # Top Right
    [1650, 480],  # Bottom Right
    [1500, 600],  # Bottom Left
]

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
    criteria=(0x01 | 0x02, 30, 0.01),  # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
)

# ── MOG2 background subtractor ──────────────────────────────────────────────
MOG2_HISTORY       = 300   # frames used to learn the background
MOG2_VAR_THRESHOLD = 50    # sensitivity — higher = less sensitive
MOG2_DETECT_SHADOWS = True

# ── Safety thresholds ───────────────────────────────────────────────────────
AREA_THRESHOLD    = 5000  # minimum debris pixel area to start the debris timer
TIME_THRESHOLD    = 3.0   # seconds debris must persist before ALARM
VEHICLE_THRESHOLD = 3.0   # seconds a vehicle must stay in ZOI before ALARM

# ── Detection stride ────────────────────────────────────────────────────────
YOLO_STRIDE = 5  # run YOLO every N frames
