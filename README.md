# USDOT Grade Crossing Monitoring

Preliminary work for real-time rail grade-crossing anomaly detection using computer vision (YOLOv8/v11, MOG2 background subtraction).

## Overview

This system focuses on improving safety at railroad grade crossings by monitoring video feeds and identifying hazards in real time, without relying on specialized sensors. Its goal is to detect situations where a vehicle becomes stuck or when debris enters the crossing, allowing alerts to be triggered and events to be recorded for later analysis.

At a high level, the system maintains a consistent view of the crossing by stabilizing a Zone of Interest (ZOI), even when the camera shifts or becomes partially obstructed. Within a specific region, it will identify known objects such as vehicles using object detection, while also monitoring for unknown objects through background-based anomaly detection. By observing how these detections behave over time, the system distinguishes between normal movement and unsafe conditions, such as a vehicle remaining stationary for too long within the crossing.

## Project Structure

- `notebooks/` — Detection pipeline notebook and helper scripts
- `scripts/` — Utility scripts
- `fine-tuning/configs/` — Committed training configs (`vehicle_finetune.yaml`)
- `docs/` — Reference documentation
- `data/` — Local dataset (gitignored — download from Releases)
- `models/` — Model weights (gitignored — auto-downloaded or fine-tuned locally)

## Dataset & Resources

The vehicle detection dataset is distributed as a pre-structured YOLO-formatted zip via the [Releases tab](https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring/releases). To set up locally:

1. Download the dataset zip from the Releases tab
2. Extract the contents into the `data/` folder at the repo root

The `data/` folder is gitignored and never committed — the release zip provides the predefined YOLO structure.

Previous test videos and a preliminary crossing dataset are also available on the Releases tab (v1.0, v1.1).

## Model

This project uses YOLOv11 Nano (`yolo11n.pt`). Base weights download automatically when running the notebook, or can be downloaded from [Ultralytics](https://docs.ultralytics.com).

## Status

This repo tracks our early-stage research progress. Current priorities:
- Fine-tune YOLO on vehicle detection
- Evaluate model generalibility across different crossing videos

## Contributors

- Ruby Gonzalez
- Gael Marquez
- David Sanchez

**Advisor:** Dr. Sergei Chuprov — UTRGV
