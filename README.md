# USDOT Grade Crossing Monitoring

Real-time rail grade-crossing anomaly detection using computer vision (YOLOv11, MOG2 background subtraction).

## Overview

This system improves safety at railroad grade crossings by monitoring video feeds and identifying hazards in real time, without relying on specialized sensors. It detects situations where a vehicle becomes stuck or debris enters the crossing, triggering alerts and recording events for later analysis.

The system stabilizes a Zone of Interest (ZOI) across the crossing even when the camera shifts or becomes partially obstructed. Within that region it identifies vehicles using YOLO object detection and monitors for unknown objects through background-based anomaly detection. By observing detections over time it distinguishes normal movement from unsafe conditions — such as a vehicle remaining stationary too long inside the crossing.

## Project Structure

```
├── pipeline/          # Core Python package (detection, safety logic, visualization)
├── notebooks/         # Interactive pipeline notebook + Jupyter kernel startup script
├── data/              # Cached downloads and video manifests (gitignored)
├── weights/           # Drop best.pt exports from Colab here (gitignored)
├── pyproject.toml     # Project metadata and runtime dependencies
├── requirements.txt   # Editable install (references pyproject.toml)
└── requirements-dev.txt  # Dev-only deps (Jupyter, ipykernel)
```

## Setup

**1. Create and activate a virtual environment (required on every new device):**

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

**2. Install dependencies:**

```bash
pip install --upgrade pip setuptools
pip install -r requirements-dev.txt
```

This installs the project in editable mode along with all runtime and dev dependencies.

**VS Code auto-install:** When the workspace is opened in VS Code, a task runs `pip install -r requirements-dev.txt` automatically. On first open, VS Code will prompt "Allow automatic tasks" — click **Allow**. Make sure your `.venv` is active (VS Code picks it up automatically if it exists at the workspace root) so the install targets the correct environment.

## Adding Videos

Videos are registered in `data/manifests/videos.yaml`. Repeat these steps for every new video before running the pipeline.

**Step 1 — Add an entry to the manifest:**

Open `data/manifests/videos.yaml` and add a new block under `videos:`:

```yaml
- id: crossing_002
  source_type: google_drive
  source: https://drive.google.com/file/d/<your-file-id>/view?usp=drive_link
  local_cache_path: data/cache/crossing_002.mp4
  vehicle_threshold: 3.0   # seconds a vehicle must dwell in the ZOI before ALARM
  time_threshold: 3.0      # seconds debris must persist in the ZOI before ALARM
```

- **`id`** — unique key for this video; also used as the name of its output folder under `outputs/runs/`
- **`source_type`** — always `google_drive` for shared Drive links; do not change
- **`source`** — paste the Google Drive share link for the video
- **`local_cache_path`** — only change the filename after `data/cache/`; convention is to match the `id` (e.g. `crossing_002` → `data/cache/crossing_002.mp4`)
- **`vehicle_threshold`** / **`time_threshold`** — alarm timing in seconds; required for every entry. Adjust based on the clip length and expected traffic pattern.

The first time you run the pipeline for a video, it downloads and caches the file automatically. Every subsequent run reuses the cached copy without re-downloading.

**Step 2 — Repeat for every new video.**

## Model Weights

Model training is done in Google Colab. After a training run, download `best.pt` and drop it into the `weights/` folder — the pipeline reads from there automatically.

`weights/` is gitignored and never committed.

## Running the Pipeline

Open `notebooks/pipeline.ipynb` and run the cells top to bottom.

- **Cell 1** — set `VIDEO_ID` to the `id` of the video you want to process
- **Cell 2** — adjust `ZOI_POINTS` to define the Zone of Interest polygon for that camera angle, then run to preview
- **Cell 3** — loads the model weights from `weights/best.pt`
- **Cell 4** — runs the full pipeline and saves all outputs to `outputs/runs/<video_id>_<timestamp>/`

## Status

Active research project. Current priorities:
- Validate detection performance across different crossing camera angles
- Improve ZOI stabilization under heavy camera movement

## Contributors

- Ruby Gonzalez
- Gael Marquez
- David Sanchez

**Advisor:** Dr. Sergei Chuprov — UTRGV
