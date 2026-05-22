# USDOT Grade Crossing Monitoring

Real-time rail grade-crossing anomaly detection using computer vision (YOLOv11, MOG2 background subtraction).

## Overview

This system improves safety at railroad grade crossings by monitoring video feeds and identifying hazards in real time, without relying on specialized sensors. It detects situations where a vehicle becomes stuck or debris enters the crossing, triggering alerts and recording events for later analysis.

The system stabilizes a Zone of Interest (ZOI) across the crossing even when the camera shifts or becomes partially obstructed. Within that region it identifies vehicles using YOLO object detection and monitors for unknown objects through background-based anomaly detection. By observing detections over time it distinguishes normal movement from unsafe conditions — such as a vehicle remaining stationary too long inside the crossing.

## Project Structure

```
├── pipeline/          # Core Python package (detection, safety logic, visualization)
├── notebooks/         # Interactive pipeline notebook + Jupyter kernel startup script
├── weights/           # Drop best.pt exports from Colab here (gitignored)
├── videos/            # Local test videos (gitignored)
├── docs/              # Reference documentation
├── run_pipeline.py    # CLI entry point
├── pyproject.toml     # Project metadata and runtime dependencies
├── requirements.txt   # Editable install (references pyproject.toml)
└── requirements-dev.txt  # Dev-only deps (Jupyter, ipykernel)
```

## Setup

```bash
pip install -r requirements-dev.txt
```

This installs the project in editable mode along with all runtime and dev dependencies. The VS Code workspace will also run this automatically on folder open.

## Model Weights

Model training is done in Google Colab. After a training run, download `best.pt` and drop it into the `weights/` folder — the pipeline reads from there automatically.

`weights/` is gitignored and never committed.

## Running the Pipeline

**Notebook:**
Open `notebooks/pipeline.ipynb`. The kernel startup script loads all imports automatically.

**CLI:**
```bash
python run_pipeline.py --video videos/<file>.mov
```

See `python run_pipeline.py --help` for all options.

## Status

Active research project. Current priorities:
- Validate detection performance across different crossing camera angles
- Improve ZOI stabilization under heavy camera movement

## Contributors

- Ruby Gonzalez
- Gael Marquez
- David Sanchez

**Advisor:** Dr. Sergei Chuprov — UTRGV
