# USDOT Grade Crossing Monitoring

Preliminary work for real-time rail grade-crossing anomaly detection using computer vision (YOLOv8/v11, MOG2 background subtraction).

## Project Structure

- `docs/` — Reference documentation and stress test findings
- `fine-tuning/` — Initial fine-tuning results (infrastructure detection) and ZOI implementation
- `notebooks/` — Detection pipeline notebooks
- `scripts/` — Python helper scripts

## Dataset & Resources
  ## Railroad crossings (object detection) > Railroad v3 - 200 images with new labels and augmentation
  - A preliminary railroad crossing dataset (static images with labels, formatted for YOLOv8) is available on the [Releases tab](https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring/releases) (v1.0)
  - Link to dataset: https://universe.roboflow.com/obstacle-detection/railroad-crossings-object-detection (Under "Dataset" tab, choose "Railroad V3")

## Video dataset
- Test videos for the detection pipeline are also available on the [Releases tab](https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring/releases) (v1.1)
- Currently exploring additional public datasets for vehicle-focused fine-tuning

## Model

This project uses YOLOv11 Nano (`yolo11n.pt`). Weights download automatically when running the notebook, or can be downloaded from [Ultralytics](https://docs.ultralytics.com).

## Status

This repo tracks our early-stage research progress. Current priorities:
- Fine-tune YOLO on vehicle detection (rather than infrastructure objects)
- Evaluate model generalizability across different crossing videos
- Source a larger public dataset
- Create a Python download script for the dataset - (FT)

## Contributors

- Ruby Gonzalez
- Gael Marquez
- David Sanchez

**Advisor:** Dr. Sergei Chuprov — UTRGV
