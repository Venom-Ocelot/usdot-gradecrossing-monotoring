# Grade Crossing Monitoring Pipeline — Development Proposal

## What We're Building

The goal of this project is a computer vision system that watches a rail grade crossing camera feed and automatically detects unsafe conditions — specifically, a vehicle that becomes stuck on the tracks or unknown debris entering the crossing zone. The system is designed to work with a standard fixed CCTV camera, no specialized sensors required.

The core pipeline works by layering three techniques together. First, we use RANSAC-stabilized optical flow to define a Zone of Interest (ZOI) that stays locked to the physical track area even if the camera shifts slightly. Second, we run MOG2 background subtraction inside that zone, which flags anything that wasn't there when the crossing was empty. Third, we use YOLO object detection to identify known vehicles, so the system can distinguish "there is a vehicle here, which is normal" from "there is something unrecognized here, which is not." The combination lets the system issue CLEAR, WARNING, or ALARM statuses in real time based on how long objects persist and whether they are recognized or not.

## The Two-Phase Development Strategy

Building a reliable system like this requires two distinct phases of work, and understanding the boundary between them is important.

**Phase 1 is about training the YOLO model.** We start from a pre-trained base model (YOLOv11 Nano) and fine-tune it on vehicle detection datasets. The base model already understands what vehicles generally look like, so fine-tuning teaches it the specific conditions relevant to our use case — overhead CCTV angles, crossing geometry, the types of vehicles that typically appear at grade crossings. Each training run produces a `best.pt` weights file and generates three metrics we track closely: box loss (how accurately the model places bounding boxes), class loss (how accurately it labels vehicle types), and dfl loss (fine-grained edge sharpness of the boxes). All three should trend downward across training epochs. The primary quality metric is mAP50, where values above 0.75 indicate a model worth testing. Our first training run achieved 0.9156 mAP50 on a CCTV vehicle dataset, which establishes a strong baseline.

The training strategy we're following is sequential chaining — each successful run's `best.pt` becomes the starting weights for the next run. This compounds improvements rather than starting from scratch each time. However, chaining only makes sense when a new run improves on the previous one. If a new dataset causes mAP50 to drop, we roll back to the last known-good weights and try a different dataset instead. We continue this loop until mAP50 stops improving meaningfully across two or three consecutive runs, which signals that we've extracted as much as we can from the available training data.

**Phase 2 is about real-world validation.** Once the model reaches a stable mAP50 threshold, we plug the `best.pt` into the full pipeline notebook and run it against actual crossing footage. This is where we find out what the model is actually wrong about in practice, which is a different question from what the training numbers suggest.

## The Gap We Need to Address

This brings us to a limitation in the current pipeline that we want to formally address. A high mAP50 score is measured against the test split of a training dataset, not against real crossing footage. The two are not the same thing. A model can perform excellently on benchmark data while still being blind to specific real-world conditions — partial occlusion by gate arms, unusual lighting, vehicle types or sizes that weren't well-represented in training. This is sometimes called training blind, and it's a problem because without a diagnostic mechanism, the only signal you get from Phase 2 is "the alarm fired" or "it didn't." You don't know whether a missed detection was because the model truly couldn't recognize the object, or whether it recognized it but with low confidence, or whether the pipeline logic needs threshold tuning.

## Proposed Addition: Diagnostic Mode

To close this gap, we're adding a diagnostic mode to the pipeline as a dedicated analysis step. Rather than replacing Cell 12 (the full pipeline run), this operates separately in Cell 13 and runs on the same video in read-only mode.

The diagnostic uses a lowered confidence threshold to surface near-miss detections — things the model almost identified but didn't commit to at normal confidence. More importantly, it computes a blind spot mask for every frame by comparing what MOG2 detected as foreground activity against what YOLO responded to. Any region where MOG2 sees significant movement but YOLO has no detection at any confidence level becomes a flagged blind spot. The cell outputs a frame-by-frame blind spot ratio, a log of near-miss detections with their confidence scores and timestamps, and visual snapshots at key moments in the video showing exactly where and when the model went silent.

The result is that Phase 2 becomes actionable. If the blind spot ratio is high and the snapshots show a specific type of object consistently going unrecognized, the next step is clear: find a dataset that contains that object type and chain a new training run targeting that specific weakness. If the blind spot ratio is low but the alarm behavior is still wrong, the issue is more likely in the safety logic thresholds, which is a tuning problem rather than a model problem. The two failure modes are now distinguishable, which makes the feedback loop between Phase 1 and Phase 2 concrete rather than speculative.
