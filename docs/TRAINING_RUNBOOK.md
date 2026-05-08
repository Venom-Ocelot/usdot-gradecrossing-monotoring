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
- **Done when:** mAP50 is consistently strong and stops improving across runs

### Phase 2 — Validation (next phase)
- **Goal:** confirm the trained model actually works on real crossing video
- **Tools:** Jupyter notebook (Cell 12 — the full pipeline)
- **What you do:** point `VIDEO_PATH` at a real crossing video, point `MODEL_PATH`
  at your best `best.pt`, run Cell 12, and observe
- **What you're looking for:** does the pipeline correctly detect stuck vehicles
  and debris? Does the alarm trigger at the right time?
- **Done when:** the pipeline performs correctly on real footage end-to-end
- **Key signals to watch for when something fails:**

  | What you see in the output | What it means | What to do |
  |---|---|---|
  | Status stays CLEAR even with a vehicle on tracks | YOLO missed the vehicle — low confidence or wrong class | Find crossing-specific detection data, chain a new run |
  | ALARM triggers immediately on every frame | MOG2 is treating the vehicle as debris instead of excluding it | YOLO confidence too low to build exclusion mask — same fix |
  | ZOI drifts off the tracks over time | Optical flow lost its reference points | Camera motion issue — not a model problem, fix in `pipeline_helpers.py` |
  | Vehicle detected but alarm never fires | Vehicle timer threshold too high for this video's scenario | Adjust `VEHICLE_THRESHOLD` in Cell 12 |
  | Alarm fires correctly but too late | Timer thresholds need tuning for this specific crossing | Adjust `TIME_THRESHOLD` / `VEHICLE_THRESHOLD` |

  The first two failures point back to Phase 1 (more training data).
  The last three are pipeline tuning issues, not model issues.

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

## Understanding the Base Weights

Before you run anything, you need to understand what `yolo11n.pt` is and why
it matters.

`yolo11n.pt` is the **base model** — a YOLO model that was already trained by
Ultralytics on millions of general internet images. It already knows what a
car looks like, what a truck looks like, what a person looks like. It is not
specialized for CCTV cameras or railroad crossings — it's a general starting
point.

**Why start from the base instead of training from scratch?**
Training a model from zero requires millions of images and days of compute.
Starting from `yolo11n.pt` means you're building on top of knowledge that
already exists. You only need a few thousand images and a few hours to teach
it the specific things your project needs. This is called fine-tuning.

**Every training run starts from either:**
- `yolo11n.pt` — the base model (fresh start, no prior fine-tuning)
- A previous `best.pt` — a model you already fine-tuned (chained run)

The difference is explained in Step 7.

---

Follow these steps every time you bring in a new dataset and retrain.
Check off each step as you go and fill in the log at the bottom.

---

## STEP 1 — Download the dataset from Roboflow (or any other source)

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
cd /path/to/usdot-gradecrossing-monotoring
unzip data/YourDatasetName.zip -d data/
```

After unzipping you should see this structure:
```
data/
├── data.yaml         ← Roboflow config file
├── train/
│   ├── images/       ← training photos
│   └── labels/       ← .txt annotation files (one per image)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

> The label files are what YOLO actually learns from. Each `.txt` file matches
> an image and contains one line per object: the class number and the bounding
> box coordinates. You never need to edit these — Roboflow generates them.

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

Replace `/YOUR/ABSOLUTE/PATH/TO/usdot-gradecrossing-monotoring` with the
output of `pwd` above.

> Why this happens: Roboflow writes paths assuming the yaml is one level above
> the train/valid/test folders. Since we extract everything into data/, the
> relative paths break. Adding the absolute `path:` fixes it permanently.
> The `path:` value must be absolute — it will differ on every machine, so
> **do not commit data.yaml** (it is already gitignored).

Also note how many classes the dataset has — you will need this for your log:
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
you will see a summary row like this:

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
training stops automatically. That is normal — it means the model has
learned as much as it can from this dataset.

**mAP50** appears at the end of each epoch during validation:
- 0.0–0.4 → model is still learning, dataset may be too small or wrong type
- 0.5–0.7 → decent, usable but keep looking for better data
- 0.7–0.9 → good, worth chaining from here
- 0.9+    → excellent

**What bad results look like and what to do:**

| What you see | What it means | What to do |
|-------------|---------------|------------|
| Losses drop then suddenly spike back up | Overfitting — model memorized training data | Find more diverse data, do not chain from this run |
| mAP50 stuck below 0.4 after 10+ epochs | Dataset is wrong type or too small | Find a better dataset, start fresh |
| mAP50 is lower than the previous run | New dataset made things worse | Do not chain from this. Roll back to previous best.pt |
| Training crashes with memory error | Batch size too large for your hardware | Reduce batch: 16 → 8 → 4 in the config |
| Losses drop but mAP50 stays low | Model learns features but wrong classes | Check that class names match what you are trying to detect |

---

## STEP 6 — After training finishes

YOLO saves two weight files:
```
models/runs/vehicle_finetune_v1/weights/
    best.pt   ← highest val mAP checkpoint — USE THIS ONE
    last.pt   ← final epoch checkpoint — ignore unless best.pt is missing
```

**Do not rename or move best.pt.** Leave it where YOLO saved it.
You will reference it by path in the next run and in the notebook.

**Backup best.pt to GitHub Releases immediately after each run.**
The `models/` folder is gitignored — weights are never committed. If you delete
the folder, do a `git clean`, or clone fresh on another machine, the file is
gone. To preserve it:
1. Go to the [Releases tab](https://github.com/ruby-gonzalez/usdot-gradecrossing-monotoring/releases)
2. Create a new release (e.g. `run-001-best`) or draft one
3. Attach `best.pt` as a release asset
4. Note the download URL in your Run Log below

This is the same pattern already used for datasets in this repo.

---

## STEP 7 — Decide what to do next

This is the most important decision in the whole process.
Compare this run's mAP50 to the previous run and ask three questions.

---

> ### ⚠️ Read this before chaining runs
>
> **Catastrophic forgetting:** When you fine-tune on a new dataset, the model
> can partially forget what it learned from the previous one. Adding a new
> dataset does not guarantee improvement — it can lower real-world performance
> even if the new mAP50 number looks fine on paper, because mAP50 only measures
> accuracy on the *new* dataset's test images, not on everything the model
> previously knew.
>
> **Training blind:** A high mAP50 on training data does not mean the model
> works on real crossing footage. You can chain run after run and never know
> what the model actually misses on a real camera until you test it. The right
> strategy is:
> 1. Validate on real crossing video first (Phase 2, Cell 12 in the notebook)
> 2. Identify exactly what the model fails on (night? partial occlusion? gates?)
> 3. Find datasets that specifically target those failures
> 4. Chain only from a known-good best.pt into that targeted data
>
> This means **Phase 2 is not the finish line — it is part of the loop.**
> You may go back and forth between Phase 1 and Phase 2 several times before
> the pipeline works reliably on real footage.

---

**1. Did mAP50 improve?**
- Yes → this dataset helped. Consider chaining from this best.pt next run.
- No  → this dataset did not help. Do not chain. Either try a different
         dataset starting fresh from yolo11n.pt, or from your last good best.pt.

**2. Is mAP50 good enough to move to Phase 2?**
- mAP50 above 0.75 and not improving across 2–3 runs → yes, move to the notebook
- Still climbing run over run → no, keep finding better data

**3. Is the model actually useful for your specific case?**
- mAP50 is a number measured on the dataset's test images
- It does not guarantee the model works on real crossing footage
- Even at 0.9+ mAP50, always validate in the notebook before calling it done

**Chaining vs fresh start — the actual decision:**

| Situation | What to do |
|-----------|------------|
| mAP50 improved, new dataset adds crossing-specific footage | Chain from this best.pt |
| mAP50 improved, but dataset is still generic | Chain anyway — improvements compound |
| mAP50 dropped compared to last run | Do NOT chain. Use previous best.pt as base |
| You want to test if a dataset helps at all (fair test) | Start fresh from yolo11n.pt |
| Adding your own labeled crossing footage | Always chain — most valuable data |

**What chaining actually means — two line changes, same command:**

Open `fine-tuning/configs/vehicle_finetune.yaml` and change these two lines:
```yaml
# Before (Run 001 — fresh start from base)
model: models/yolo11n.pt
name:  vehicle_finetune_v1

# After (Run 002 — chained, builds on Run 001)
model: models/runs/vehicle_finetune_v1/weights/best.pt
name:  vehicle_finetune_v2
```

Then run the exact same command as always:
```bash
yolo train cfg=fine-tuning/configs/vehicle_finetune.yaml
```

YOLO loads best.pt instead of the base weights and starts learning from where
Run 001 left off. Everything else stays the same.

Your models/runs/ folder becomes a history you can always roll back to:
```
models/runs/
├── vehicle_finetune_v1/weights/best.pt  ← Run 001
├── vehicle_finetune_v2/weights/best.pt  ← Run 002, chained from v1
└── vehicle_finetune_v3/weights/best.pt  ← Run 003, chained from v2
```

If Run 003 mAP50 drops below Run 002, point `model:` back at v2's best.pt.
That version is your new starting point.

**When to stop training and move to Phase 2 (the notebook):**
- mAP50 is above 0.75 and has not improved across the last 2–3 runs
- You have run at least one dataset that includes crossing-specific footage
- You have a best.pt you are confident in

At that point, update Cell 1b in the notebook:
```python
MODEL_PATH = "models/runs/vehicle_finetune_v2/weights/best.pt"
```
And move to Phase 2 validation on a real crossing video.

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
         mAP50 stopped improving → move to Phase 2, test in the notebook
```

---

### Run 001
| Field | Value |
|-------|-------|
| Date | 2026-04-05 |
| Dataset | CCTV_Vehicles v7 (Roboflow — atledtech) |
| Dataset size | 5,758 train / 549 val / 284 test |
| Classes | Car, Jeep, Motorcycle, Tricycle, Truck, Van (nc=6) |
| Base model | yolo11n.pt (fresh start) |
| Epochs run | 14 of 50 (early stopping) |
| Final box_loss | 0.6149 |
| Final cls_loss | 0.5829 |
| Final dfl_loss | 0.9563 |
| Final mAP50 | 0.9156 |
| Stopped early? | Yes — no improvement after epoch 14 |
| Weights saved to | models/runs/vehicle_finetune_v1/weights/best.pt |
| Decision for next run | Chain from this best.pt — excellent result, add crossing-specific data |
| Notes | Excellent result for a first run. Generic CCTV overhead vehicle dataset, no crossing-specific footage. High mAP50 likely because dataset angle matches crossing camera perspective well. |

---

### Run 002
| Field | Value |
|-------|-------|
| Date | |
| Dataset | |
| Dataset size | |
| Classes | |
| Base model | ← fresh (yolo11n.pt) or chained from Run 001 best.pt? |
| Epochs run | |
| Final box_loss | |
| Final cls_loss | |
| Final dfl_loss | |
| Final mAP50 | |
| Improved over Run 001? | ← yes / no |
| Stopped early? | |
| Weights saved to | |
| Decision for next run | ← chain / fresh / move to Phase 2 |
| Notes | |

---

### Run 003
| Field | Value |
|-------|-------|
| Date | |
| Dataset | |
| Dataset size | |
| Classes | |
| Base model | ← fresh (yolo11n.pt) or chained from Run 002 best.pt? |
| Epochs run | |
| Final box_loss | |
| Final cls_loss | |
| Final dfl_loss | |
| Final mAP50 | |
| Improved over previous run? | ← yes / no |
| Stopped early? | |
| Weights saved to | |
| Decision for next run | ← chain / fresh / move to Phase 2 |
| Notes | |
