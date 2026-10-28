# YOLO + MediaPipe: separate alternative

This adds an independently trained alternative while preserving the existing models and baseline notebooks. It does not replace the chosen model or claim 95% accuracy. It uses **one final back/side/prone classifier**, with YOLO and MediaPipe providing measurements.

The [completed original-data experiment](../artifacts/hybrid_alternative_v1/README.md) contains both actual trained exports, test/validation reports, features, annotated examples and the hybrid deployment ZIP. On 882 matching test images, accuracy increased from 23.5% to 29.1%; across 4,421 eligible publisher-labeled images, pose coverage increased from 55.8% to 71.1%. The manually reviewed Colab ZIP was not available here, and no 95% or Pi-performance claim is established.

## Workflow

1. Run the selected MediaPipe pose/face extractor on the original frame as before.
2. If the pose is usable, retain that observation. Multiple detected people/faces remain unknown.
3. Otherwise, run YOLO pose to find a body. Reject multiple body detections. Expand the single body box by 20% and retry MediaPipe on the crop.
4. Remap crop landmarks into the original 640×640 coordinate system before computing geometry and drawing the box.
5. If cropping still fails, the default alternative can use YOLO's 17 body keypoints plus MediaPipe dense face landmarks from the body region. It retains the original torso/visibility threshold values; these are research settings, not validated infant confidence thresholds.
6. Feed the resulting 12 entries, with missing values represented explicitly, to one independently trained posture classifier. Use the existing posture risk mapping, exact-frame overlay, TCP 5000 alarm policy and stale-frame handling.

Set keypoint_fallback=False to test the stricter crop-only alternative. Cropping alone recovered no usable pose in a six-example integration smoke test; that tiny selected sample is not an accuracy estimate. The keypoint fallback is therefore important to evaluate, rather than assuming that cropping fixes infant pose detection.

## The 12-metric contract

| Metrics | Hybrid interpretation |
| --- | --- |
| Shoulder/hip widths, shoulder/hip angles, angle difference (5) | Same normalized image geometry, from MediaPipe when usable or remapped YOLO keypoints on fallback. |
| Four shoulder/hip visibility entries (4) | MediaPipe visibility on MediaPipe frames; native YOLO keypoint confidence on YOLO-fallback frames. These scores are not interchangeable calibrated probabilities. This is an explicitly changed feature definition and requires the separate classifier. |
| face_detected, face_landmark_count (2) | Actual MediaPipe dense face landmark output. YOLO face/body boxes are never counted as face landmarks. |
| head_lift_ratio (1) | Original MediaPipe world-z proxy when available. Missing on YOLO-only body observations; 2D keypoints cannot measure it. Imputation/missing indicators are fitted on training rows. No depth or airway clearance is invented. |

The alternative retains the 12 named inputs but does not pretend to have all 12 valid measurements on every frame. If the requirement is specifically MediaPipe visibility and a measured depth value on every image, YOLO-only fallback does not satisfy that stricter requirement. Crop-only mode preserves MediaPipe-only measurement semantics but retains its failures. Classifier metadata records the adapter, thresholds, weights hash, and pose/face asset hashes; do not feed hybrid measurements into an older classifier. Training rejects mixed baseline/hybrid CSVs.

## Open the alternative Colab

[Open hybrid_mediapipe_yolo_colab.ipynb](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/hybrid_mediapipe_yolo_colab.ipynb).

1. Use a CPU runtime. Run sections 1–2 to get source and install the separate hybrid dependencies. Torch installation can take time; no Git LFS download is needed for model setup.
2. Upload the **same sorted BabyTraining.zip** in section 3; import it in section 4. No additional sorting is necessary.
3. Run section 5. It separately trains a matching baseline and the hybrid using the same labels, source groups and seed, with validation-only selection among seven classifier settings.
4. Section 6 prints both test results, prone recall, unknown rejection counts, usable YOLO recoveries and latency on this machine. Test results describe two preplanned alternatives; the validation report determines the recommendation. This pilot does not establish a fresh unseen-infant result.
5. Section 7 displays hybrid sample predictions with boxes and risk levels. Section 8 downloads SIDS_Pi5_Hybrid_Alternative.zip. Download before resetting Colab. The old model in its original run is untouched.

## Reuse an already open Colab session

After current execution stops, pull source with Git LFS smudging disabled as shown in [IMPROVE_MODEL.md](IMPROVE_MODEL.md). Install the following in new code cells; the base requirements from section 2 must already be installed:

```python
%pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1 torchvision==0.24.1
%pip install -r /content/SeniorDesignProject/training/requirements-hybrid.txt
%pip install --no-deps ultralytics==8.3.200
```

Then use this cell. Keep BASELINE_FEATURES and HYBRID_RUN unchanged to resume the same experiment. If labels/configuration change, choose a new output directory. Do not restart a runtime containing your only local upload/results.

```python
import importlib
from pathlib import Path
for name in ['assets', 'vision', 'hybrid_vision', 'data', 'learning', 'optimize', 'hybrid_experiment']:
    importlib.reload(importlib.import_module('cribwatch.' + name))
from cribwatch.hybrid_experiment import run_hybrid_comparison
BASELINE_FEATURES = globals().get('BASELINE_FEATURES', FEATURES)
HYBRID_RUN = Path(RUN) / 'hybrid-alternative-v1'
hybrid = run_hybrid_comparison(
    MANIFEST, HYBRID_RUN, ROOT / 'models',
    existing_features=BASELINE_FEATURES, pose_variant='lite',
)
MODEL = Path(hybrid['model_dir'])
FEATURES = Path(hybrid['features'])
print('Hybrid model:', MODEL)
print('Matching baseline:', hybrid['baseline_model_dir'])
print('Comparison:', hybrid['comparison_path'])
```

The existing showcase section can use this MODEL. For the ZIP, set ZIP_OUT to RUN / 'SIDS_Pi5_Hybrid_Alternative.zip' so it does not overwrite a previous deployment ZIP. The builder detects the adapter and includes the required extra code, weights and requirements. Full pose can be tried using pose_variant='full' in a NEW output directory; the default Lite comparison minimizes download/compute while isolating YOLO's contribution.

## Weights and future infant tuning

The bundled default is **official generic yolov8n-pose**, not an infant-tuned detector and not the reference repository's weights. Its URL/hash and AGPL attribution/license are in models/YOLO_ATTRIBUTION.md and models/YOLO_LICENSE.txt. No external reference implementation was copied.

run_hybrid_comparison also accepts yolo_weights='/path/to/trusted-infant-pose.pt'. This must be a compatible YOLO COCO-17 pose checkpoint with trustworthy provenance and permitted use/redistribution. The exact file is copied under its own basename and bound by checksum. Unknown keypoint layouts fail explicitly. Do not infer baby-specific capability from a renamed generic checkpoint. Infant fine-tuning needs infant body-keypoint annotations; posture folders alone are not keypoint ground truth. Train any detector/embedding using training groups only; never fine-tune on the final evaluation infants.

## Pi integration

The deployment ZIP still contains one .joblib posture classifier, the selected MediaPipe pose asset, face asset, and additionally the YOLO .pt feature extractor. Read the bundled README.md and PROTOCOL.md for the same Pi 5 camera/Pi 4 TCP alarm setup. Install the base requirements in a separate Pi environment, then:

```bash
python -m pip install -r requirements-hybrid.txt
python -m pip install --no-deps ultralytics==8.3.200
```

We intentionally use the existing opencv-contrib-python distribution rather than install a second opencv-python package through Ultralytics. Its dependency metadata names opencv-python, so pip check may flag that package name despite cv2 being supplied by contrib; verify the pinned cv2 import and run the offline sample check. All other direct hybrid dependencies are installed explicitly. Compatible ARM64/Python 3.13 Torch/torchvision wheels were found on PyPI; actual Pi imports/performance are not verified here.

Inference currently uses CPU. It is slower on fallback frames because it can run full-frame MediaPipe, YOLO, cropped MediaPipe and face extraction. Measure real Pi frame age against the existing stale-frame/alarm limits. Do not claim live performance from Colab timings or increase stale limits merely to conceal slow inference. A separate deployable alternative remains useful even if it is less accurate; retain its honest report.
