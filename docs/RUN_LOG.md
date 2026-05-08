# Pipeline Run Log

Tracks each inference run against the test video.  
**Baseline = Run 001 (pretrained COCO).** All subsequent runs are compared against it.

---

## Run 001 — Pretrained Baseline

| Field | Value |
|---|---|
| Date | 2026-04-09 |
| Model | `yolo11n.pt` (YOLOv11n, pretrained on COCO — no fine-tuning) |
| Weights source | Ultralytics auto-download |
| Video | `Rail-Grade-Crossing-StressTest.mov` |
| Resolution | 1772 × 872 |
| Total frames | 1027 (1015 processed in full pipeline pass) |
| FPS | 31.45 |

### Video Metadata
| Field | Value |
|---|---|
| Duration | ~32.5 s |
| ZOI polygon | TL (100,400) · TR (500,350) · BR (1650,480) · BL (1500,600) |
| ZOI area (pixels) | 155,276 px |

### Optical Flow / ZOI Stabilization
| Metric | Value |
|---|---|
| Feature points detected | 200 |
| Points successfully tracked (frame 1→2) | 199 / 200 |
| RANSAC inliers (background) | 180 |
| RANSAC outliers (moving objects) | 19 |

### YOLO Detections — Static Test Frame (~14s)
> Single-frame snapshot used for static evaluation (gate-damaged scene, SUV still on tracks).

| Class | Confidence |
|---|---|
| truck | 0.60 |
| car | 0.35 |

### Active Exclusion Masking — Static Frame (~14s)
| Metric | Value |
|---|---|
| Anomalous pixels inside ZOI (after YOLO subtraction) | 696 px |
| ZOI total pixels | 155,276 px |
| Debris coverage | 0.45% |

### Safety Logic Evaluation — Static Frame (~14s)
| Parameter | Value |
|---|---|
| Area threshold | 5,000 px |
| Debris time threshold | 3.0 s |
| Vehicle time threshold | 5.0 s |
| Vehicle detected in ZOI | Yes (`truck`) |
| **STATUS** | **WARNING** |

### Full Pipeline Pass (Cell 12 — ZOI + MOG2 + YOLO + Safety Logic)
| Snapshot | Result |
|---|---|
| 0s — SUV entering as gates come down | Captured |
| 7s — SUV fully trapped on tracks | Captured |
| 14s — Gate damaged, SUV still on tracks | Captured |
| 28s — Train entering frame | Captured |
| Frames processed | 1,015 |

### Notes
- `yolo11n.pt` was not fine-tuned on grade crossing data. It detects generic COCO vehicles.
- These values are the **comparison baseline** for all subsequent fine-tuned runs.

---

## Run 002 — Fine-Tuned

| Field | Value |
|---|---|
| Date | 2026-04-09 |
| Model | `runs/detect/models/vehicle_finetune_v1/weights/best.pt` |
| Trained from | `yolo11n.pt` + CCTV_Vehicles v7 (Roboflow) |
| Training result | mAP50=0.9156, early stop epoch 14/50 |
| Training config | `fine-tuning/configs/vehicle_finetune.yaml` |
| Video | `Rail-Grade-Crossing-StressTest.mov` |
| Resolution | 1772 × 872 |
| Total frames | 1027 (1015 processed in full pipeline pass) |
| FPS | 31.45 |

### YOLO Detections — Static Test Frame (~14s)

| Class | Confidence |
|---|---|
| Car | **0.84** |
| Car | 0.27 |

### Active Exclusion Masking — Static Frame (~14s)
| Metric | Value | vs Run 001 |
|---|---|---|
| Anomalous pixels inside ZOI (after YOLO subtraction) | 493 px | -29% |
| ZOI total pixels | 155,276 px | — |
| Debris coverage | 0.32% | -29% |

### Safety Logic Evaluation — Static Frame (~14s)
| Parameter | Value |
|---|---|
| Area threshold | 5,000 px |
| Debris time threshold | 3.0 s |
| Vehicle time threshold | 5.0 s |
| Vehicle detected in ZOI | Yes (`Car`, `Car`) |
| **STATUS** | **WARNING** |

### Full Pipeline Pass (Cell 12 — ZOI + MOG2 + YOLO + Safety Logic)
| Snapshot | Result |
|---|---|
| 0s — SUV entering as gates come down | Captured |
| 7s — SUV fully trapped on tracks | Captured |
| 14s — Gate damaged, SUV still on tracks | Captured |
| 28s — Train entering frame | Captured |
| Frames processed | 1,015 |

### Notes
- Fine-tuned model uses domain-specific class label `Car` instead of COCO's generic `truck`/`car`.
- Primary detection confidence jumped from 0.60 → **0.84** on the ~14s test frame.
- Debris pixels reduced by 29%, meaning the fine-tuned model covers more of what MOG2 flags as foreground, reducing false alarm noise.

---

## Comparison Summary

| Metric | Run 001 (pretrained) | Run 002 (fine-tuned) | Δ |
|---|---|---|---|
| Primary detection conf (14s) | 0.60 (`truck`) | **0.84** (`Car`) | +40% |
| Debris pixels after masking | 696 px | **493 px** | -29% |
| Debris coverage | 0.45% | **0.32%** | -29% |
