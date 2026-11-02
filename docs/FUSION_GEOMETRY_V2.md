# Measured geometry, YOLOv8 + MediaPipe, and one posture network

This is a separate model-development workflow. The [uploaded 31.5% team package](TEAM_TEST_RUNS.md) and older models remain available. The target is **0.95 accuracy (95%)**, counting unknown readings as errors, with strong recall for every posture. Neither changing the score label nor drawing a skeleton establishes that target.

## What is changing

The original model used only 12 metrics. Distances discard direction, and schema-2 shoulder/hip angles fold a line and its 180-degree reversal together. A back and a belly view can therefore have similar numerical features. Missing face/3D measurements make that ambiguity worse. High landmark visibility is a model estimate, not proof that anatomical joints were identified correctly. Plastic dolls also differ substantially from the human images used by generic pose/face models.

The new alternative keeps all **12 raw metrics as explicit network inputs**, with training-only normalization, individual missing-value indicators and a pose-quality flag. A small visual encoder supplements them with appearance information. Their features enter **one final three-class posture classifier**. This is not an image-only replacement for the metrics. Training sometimes suppresses visual features to encourage use of the metric branch. Test reports include branch ablations to quantify sensitivity to the metrics and to the image features; these are not independent clinical validations.

YOLOv8 provides body localization and optional 2D keypoint recovery. MediaPipe provides pose/face measurements and its depth proxy when available. YOLO-fallback visibility entries are native keypoint confidence, not calibrated MediaPipe visibility. Missing head depth is never replaced by an invented number. No body evidence or multiple detected people/faces stays unknown. A usable image can supplement poor body geometry, but an empty/undetected scene is not accepted just because an image classifier emits a score.

## Two explicit metric contracts

- **Schema 2:** the original projected widths and axial image-line angles; old classifiers retain it.
- **Schema 3:** the same 12 named inputs, but shoulder/hip angles are directed angles between the anatomical left-to-right transverse vector and the shoulder-midpoint-to-hip-midpoint torso vector. In 2D, theta = atan2(cross(torso, transverse), dot(torso, transverse)). Their difference is a circular directed-angle difference. This retains facing-direction information and is invariant to rotating the entire image in its plane.

Schema 3 requires fresh extraction and retraining. It does not relabel old vectors. Correct anatomical left/right landmarks are still essential: a detector that swaps shoulders can produce a wrong facing cue. **These projected angles are not physical roll angle, head elevation, airway clearance, or a medical-risk percentage.** Measuring roll in degrees requires calibration and reference angle labels or suitable validated 3D sensing. Do not manufacture roll labels by assigning all back/side/prone images 0/90/180 degrees.

## Colab

Open [fusion_geometry_colab.ipynb](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/fusion_geometry_colab.ipynb). For faster fine tuning select a GPU runtime before setup. The notebook keeps Colab's matching Torch/TorchVision pair; Pi inference uses the bundled pinned CPU-compatible versions.

1. Run setup, upload the same reviewed BabyTraining.zip and import its folders. Keep work local and download results before resetting.
2. Extract schema-3 features using the versioned YOLOv8/MediaPipe adapter. Inspect pose coverage and failures. Keep the source camera view/framing consistent with capture; the actual reported test cohort includes detection failures.
3. Train one network. Every learned stage fits training groups only: image fine tuning, metric normalization, and missing-value processing. Epoch, calibration temperature and acceptance threshold are selected using validation data only. The test split is evaluated afterward. Color augmentation does not alter joint coordinates; geometric image flips/crops are not applied without transforming the associated metrics.
4. Read overall accuracy, every class's recall, confusion matrix, coverage, detector-evidence ceiling and the metric/image ablations. This publicly inspected pilot is exploratory; use a fresh cohort with real infant/session IDs for a final unseen-infant claim. A doll-only score is not infant accuracy.
5. Display example scans and export a new deployment ZIP. It contains one posture network, its model/feature metadata, exact YOLO/MediaPipe assets, code and TCP alarm support. Do not mix this model with an older feature implementation.

The original 12-metric component remains visible in code, saved feature tables and the display.

## What the camera display shows

The live/offline display draws actual visible joints, shoulder/hip vectors, torso connections, shoulder/hip angles and an actual dense-face region when available. It includes all 12 metrics, all three posture scores, per-image confidence, saved overall test accuracy, posture risk, alarm state and link state. Missing/poor joints do not get replaced by a template skeleton. The coordinates are attached to the exact classified frame, including crop remapping. Unknown remains unknown.

Use the existing Pi 5 command in a NEW unpacked deployment folder, after installing base and hybrid requirements as described in the bundle:

```bash
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-risk moderate --log test-runs/fusion-001.jsonl --record test-runs/fusion-001.mp4
```

The Pi 4 receiver still uses **TCP 5000**, shared-key provisioning and the existing dwell/latching/fault policy. Add --alarm-host with the Pi 4 address only after configuring its bundled receiver. Geometry stays in the local display/log; alarm packets omit bulky drawing data to preserve the message-size limit. Camera/inference faults are not suppressed to improve a score. Pi throughput, sound and fault timing must be measured on the actual hardware.

For SSH/headless operation, omit --display and review the recorded annotated MP4 on your laptop. A still-image test shows posture/risk but does not run the live dwell timer or send a physical alarm; actual sound must be tested through the live monitor and configured Pi 4 receiver.

An optional normalized --roi x1,y1,x2,y2 crops the crib before classification and drawing. Use the same ROI while collecting new data. This can exclude an operator outside the crib, but it must not crop away the baby or be changed between training and evaluation without documenting the view change.

## Your next action: record the actual overhead doll view

The three provided terminal screenshots do not contain the camera frames, so they cannot establish whether coordinates were anatomically correct. The custom FINAL TEST SUMMARY script is also not included in the uploaded ZIP; its always-true monitor-fault flag cannot be diagnosed from its printed values alone.

Use the current team package's --record/--log options to save short static back/side/prone trials with the true label in the filename. Keep each video and JSONL together and preserve failures. Capture without hands or another person in the crib region. Send the actual clips and the custom test script if its fault flag persists. The fixed camera is overhead, as the user confirmed.

For labeled image collection, the updated source/bundle also includes:

```bash
PYTHONPATH=src python -m cribwatch.capture --picamera --preview \
  --output BabyTraining --subject-kind doll --subject-id doll01 \
  --session-id overhead_lightingA_day1 --posture prone --seconds 30 --fps 1
```

Repeat with back and side while the doll stays in the specified posture throughout each capture. Use the same real session ID for clips from the same collection setting. Use new real session IDs for separate recording sessions, rather than inventing IDs per frame. The tool writes class folders and metadata.csv; dolls never receive fabricated infant IDs. Review labels before making a ZIP. Vary genuine recording conditions, camera framing, clothing and doll orientation; hundreds of adjacent frames from one trial do not establish independence. The existing grouped-split guard needs adequate groups in every class/fold, so add varied sessions rather than bypassing it.

## What is still needed for a robust 95% result

If detector-evidence coverage is below 95%, no classifier using that gate can reach 95% all-image accuracy on that cohort. The next detector improvement needs verified infant/doll body/keypoint annotations; posture folders alone do not provide shoulder/hip ground truth for YOLO pose fine tuning. The suggested reference project used infant-tuned extractors; the bundled generic YOLO checkpoint is not equivalent. Do not train another extractor on test infants and then call the downstream classifier test independent.

There is now a concrete annotation/fine-tuning route for those missing coordinates. On a desktop with the reviewed images and an imported images.csv, run:

```bash
PYTHONPATH=src python -m cribwatch.annotate_torso folder-audit/images.csv --output verified-torso.jsonl
```

Drag a real body bounding box, then click anatomical left shoulder, right shoulder, left hip and right hip. Mark a point missing with x if its location cannot be established; do not guess invisible joints. Save with s, skip with n, quit with q. Each saved annotation is durable and skipped on the next run. This is deliberately human-reviewed ground truth, not automatic conversion of failed model predictions into labels.

Use the doll's anatomical left/right, not the screen's left/right. Do not swap labels when it turns over. The camera overlay labels these anchors LS, RS, LH and RH so a confidently misplaced or reversed joint can be spotted.

Build and fine-tune a four-keypoint YOLOv8 model in a GPU-enabled Colab using that same existing posture experiment's split assignments:

```bash
PYTHONPATH=src python -m cribwatch.torso_pose verified-torso.jsonl \
  --output torso4-data --split-manifest existing-model/split_manifest.csv --train --epochs 50
```

Keep the same source images and group IDs. The supplied split manifest must cover all annotated images; annotations from test groups remain in the test folder and are not used by YOLO training/selection. The exporter rejects groups crossing folds. If no split manifest is supplied, it creates grouped splits for a new annotation cohort; do not assume those match a different posture experiment automatically.

Use the resulting best.pt under a distinct name, such as doll_torso4.pt, and set the adapter keypoint_layout to torso4 in a NEW extraction/training run. The order is explicitly [left shoulder, right shoulder, left hip, right hip]. The hybrid runtime supports this layout as well as the old COCO-17 checkpoint. Fine-tuning a pose extractor does not produce a new posture classifier by itself: re-extract the 12 metrics with that exact detector, retrain the one final posture network, and evaluate its untouched test groups. This route is implemented; no four-keypoint training result is claimed before verified annotations exist.

The notebook selects strategy=yolo_primary for this explicitly supplied fine-tuned torso checkpoint. This matters because the original fallback-only design never asked YOLO to correct a confidently wrong MediaPipe pose. In primary mode a usable trained YOLO torso takes priority; MediaPipe supplies face observations and an associated depth proxy when genuinely available. If the primary keypoints are unusable, usable MediaPipe remains a fallback. Multiple detected people/faces stay unknown. The bundled generic checkpoint keeps the old fallback strategy so existing model measurements and reported scores are preserved; do not silently switch strategy without re-extracting and retraining.

Collect representative overhead doll data for the showcase, independently review labels and landmark correctness, and retain separate recordings for evaluation. For infant claims, use an independent permitted infant cohort with real grouping. Report prone recall, falsely reassuring back predictions, failures, continuous alarm latency and false alarms/hour. The risk categories remain experimental posture proxies; visible face/head estimates do not establish an unobstructed airway or justify automatically suppressing a prone alert.
