# Training Runbook
## How to add a new dataset and fine-tune the YOLO model

---

## Project Scope — Read This First

This project has two distinct phases. Understanding which phase you are in
determines what tools you use and what you should be doing.

### Phase 1 — Training (where we are now)
- **Goal:** build a YOLO model that reliably detects vehicles in CCTV footage
- **Tools:** terminal only — `yolo train` command
- **What you do:** find datasets → extract → train → log results → repeat
- **Jupyter notebook:** not used at all during this phase
- **Done when:** mAP50 is consistently strong across multiple datasets/runs

### Phase 2 — Validation (next phase)
- **Goal:** confirm the trained model actually works on real crossing video
- **Tools:** Jupyter notebook (Cell 12 — the full pipeline)
- **What you do:** point `VIDEO_PATH` at a real crossing video, point `MODEL_PATH`
  at your best `best.pt`, run Cell 12, and observe
- **What you're looking for:** does the pipeline correctly detect stuck vehicles
  and debris? Does the alarm trigger at the right time?
- **Done when:** the pipeline performs correctly on real footage end-to-end

### The connection between phases
```
Phase 1 (training)     →    produces best.pt
Phase 2 (validation)   →    plugs best.pt into the notebook and tests on real video
```
If Phase 2 reveals the model is still missing things, you go back to Phase 1,
find better data, retrain, and test again. This loop continues until the
pipeline works reliably on real crossing footage.

### What each component does
```
Dataset images      →  teach YOLO what vehicles look like (training only)
best.pt             →  the model's learned knowledge, used by the notebook
Video file          →  what the notebook actually watches and analyzes
RANSAC + Opt. Flow  →  keeps the ZOI locked to the tracks (notebook, Cell 12)
MOG2                →  detects unknown objects/debris (notebook, Cell 12)
YOLO                →  detects known vehicles (notebook, Cell 12, uses best.pt)
Safety logic        →  decides when to trigger an alarm (notebook, Cell 12)
```

---

Follow these steps every time you bring in a new dataset and retrain.
Check off each step as you go and fill in the log at the bottom.

---

## STEP 1 — Download the dataset from Roboflow

1. Find your dataset
2. Click **"Download Dataset"**
3. Select format: **YOLOv11**
4. Download the ZIP file

> What to look for in a good dataset:
> - 500–5,000 images 
> - Fixed camera angle (overhead/side), not dashcam
> - Classes relevant to your use case (Car, Truck, Pedestrian, etc.)
> - Pre-split into train / val / test sets

---

## STEP 2 — Put the ZIP in the data/ folder and extract it

Move the downloaded ZIP into:
```
data/
```

Then open a terminal, navigate to the project root, and run:
```bash
cd /Users/davidsanchez/Desktop/usdot-gradecrossing-monotoring
unzip data/YourDatasetName.zip -d data/
```

After unzipping you should see this structure:
```
data/
├── data.yaml         ← Roboflow (or any other source) config file
├── train/
│   ├── images/       ← training photos
│   └── labels/       ← .txt annotation files
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## STEP 3 — Fix data.yaml (always broken from Roboflow by default)

Open `data/data.yaml`. It will look like this by default:
```yaml
train: ../train/images    ← WRONG, broken path
val: ../valid/images      ← WRONG
test: ../test/images      ← WRONG
```

Fix it by adding a `path:` line at the top and changing the three paths.
First, get your repo root path by running this from the project root:
```bash
pwd
```

Then update `data/data.yaml`:
```yaml
path: /YOUR/ABSOLUTE/PATH/TO/usdot-gradecrossing-monotoring/data
train: train/images
val: valid/images
test: test/images
```

Replace `/YOUR/ABSOLUTE/PATH/TO/usdot-gradecrossing-monotoring` with the output of `pwd` above.

> Why this happens: Roboflow writes paths assuming the yaml is one level above
> the train/valid/test folders. Since we extract everything into data/, the
> relative paths break. Adding the absolute `path:` fixes it permanently.
> The `path:` value must be absolute — it will differ on every machine, so
> **do not commit data.yaml** (it is already gitignored).

Also note how many classes the dataset has — you'll need this for your log:
```yaml
nc: 6
names: ['Car', 'Jeep', 'Motorcycle', 'Tricycle', 'Truck', 'Van']
```

---

## STEP 4 — Run training

From the project root, run:
```bash
cd /path/to/usdot-gradecrossing-monotoring
yolo train cfg=fine-tuning/configs/vehicle_finetune.yaml
```

> If `yolo` is not on your PATH, activate your virtual environment first:
> ```bash
> source /path/to/your/venv/bin/activate
> ```
> Then run the `yolo train` command above.

> **Hardware note:** The config auto-detects the best available device.
> No changes needed regardless of who is running it or what machine they're on:
> - NVIDIA GPU (CUDA) → used automatically on Linux/Windows
> - Apple Silicon (MPS) → used automatically on Mac M-series
> - CPU only → fallback if no GPU found, slowest but always works
>
> If you get an out-of-memory error on CPU, open `fine-tuning/configs/vehicle_finetune.yaml`
> and change `batch: 16` to `batch: 8` or `batch: 4`.

Training will start and print one line per batch. At the end of each epoch
you'll see a summary row like this:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
  1/50    4.26G      1.174      3.874      1.203        67       640
  2/50    4.26G      1.089      3.201      1.178        71       640
  3/50    4.27G      1.031      2.887      1.154        58       640
```

---

## STEP 5 — Monitor the losses

Watch these three numbers trend **downward** across epochs:

| Metric | What it measures | Good sign |
|--------|-----------------|-----------|
| `box_loss` | Accuracy of bounding box position | Dropping each epoch |
| `cls_loss` | Accuracy of class label (Car vs Truck) | Dropping each epoch |
| `dfl_loss` | Fine-grained box edge sharpness | Dropping each epoch |

**Early stopping:** if none of the losses improve for 10 epochs in a row,
training stops automatically. That's normal — it means the model has
learned as much as it can from this dataset.

**mAP50** appears at the end of each epoch during validation:
- 0.0–0.4 → model is still learning
- 0.5–0.7 → decent, usable
- 0.7–0.9 → good
- 0.9+    → excellent (may be overfitting, check val vs train gap)

---

## STEP 6 — After training finishes

YOLO saves two weight files:
```
models/runs/vehicle_finetune_v1/weights/
    best.pt   ← highest val mAP checkpoint — USE THIS ONE
    last.pt   ← final epoch checkpoint — ignore unless best.pt is missing
```

Open the notebook and update **Cell 1b**:
```python
# Before (base model)
MODEL_PATH = "yolo11n.pt"

# After (your fine-tuned model)
MODEL_PATH = "models/runs/vehicle_finetune_v1/weights/best.pt"
```

That's it. The entire pipeline now runs with your fine-tuned detector.

---

## STEP 7 — Before starting the next run

**Always increment the run name** in `fine-tuning/configs/vehicle_finetune.yaml`
so you never overwrite a previous result:
```yaml
name: vehicle_finetune_v2   ← change this each run (v1, v2, v3...)
```

**Decide whether to chain or start fresh:**

| Situation | Set `model:` to |
|-----------|----------------|
| Testing if a new dataset is useful (fair comparison) | `models/yolo11n.pt` |
| Adding crossing-specific data on top of what was learned | `models/runs/vehicle_finetune_v1/weights/best.pt` |
| Your own labeled footage (most valuable data) | previous run's `best.pt` |
| Previous run's mAP was worse — rolling back | the last good `best.pt` |

Chaining example — Round 2 builds on Round 1:
```yaml
model: models/runs/vehicle_finetune_v1/weights/best.pt
name:  vehicle_finetune_v2
```

Your `models/runs/` folder becomes a history you can always roll back to:
```
models/runs/
├── vehicle_finetune_v1/weights/best.pt  ← Round 1
├── vehicle_finetune_v2/weights/best.pt  ← Round 2, chained from v1
└── vehicle_finetune_v3/weights/best.pt  ← Round 3, chained from v2
```

---

## RUN LOG

Fill this in every time you complete a training run.

The goal of this log is twofold:
1. Track whether each new dataset actually improved the model (compare mAP50)
2. Build a record you can reference when writing the project report

What good progress looks like over time:
```
Run 001  mAP50: 0.61  (generic CCTV dataset — baseline, no crossing footage)
Run 002  mAP50: 0.74  (crossing-specific data added — clear improvement)
Run 003  mAP50: 0.73  (different dataset, no gain — not worth chaining)
Run 004  mAP50: 0.81  (your own labeled footage — biggest jump, most valuable)
         ↓
         Ready for Phase 2 — test on real crossing video in the notebook
```

When mAP50 stops improving across multiple runs, that is your signal to move
to Phase 2 and validate in the Jupyter notebook with a real crossing video.

---

### Run 001
| Field | Value |
|-------|-------|
| Date | 2026-04-05 |
| Dataset | CCTV_Vehicles v7 (Roboflow — atledtech) |
| Dataset size | 5,758 train / 549 val / 284 test |
| Classes | Car, Jeep, Motorcycle, Tricycle, Truck, Van (nc=6) |
| Base model | yolo11n.pt |
| Epochs run | _fill in when done_ |
| Final box_loss | _fill in_ |
| Final cls_loss | _fill in_ |
| Final dfl_loss | _fill in_ |
| Final mAP50 | _fill in_ |
| Stopped early? | Yes / No |
| Weights saved to | models/runs/vehicle_finetune_v1/weights/best.pt |
| Notes | First fine-tuning run. CCTV overhead vehicle dataset. MPS (Apple M5). |

---

### Run 002
| Field | Value |
|-------|-------|
| Date | |
| Dataset | |
| Dataset size | |
| Classes | |
| Base model | ← fresh (yolo11n.pt) or chained (v1/best.pt)? |
| Epochs run | |
| Final box_loss | |
| Final cls_loss | |
| Final dfl_loss | |
| Final mAP50 | |
| Stopped early? | |
| Weights saved to | |
| Notes | |

---

### Run 003
| Field | Value |
|-------|-------|
| Date | |
| Dataset | |
| Dataset size | |
| Classes | |
| Base model | ← fresh (yolo11n.pt) or chained (v2/best.pt)? |
| Epochs run | |
| Final box_loss | |
| Final cls_loss | |
| Final dfl_loss | |
| Final mAP50 | |
| Stopped early? | |
| Weights saved to | |
| Notes | |
