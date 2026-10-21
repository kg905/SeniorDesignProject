# CribWatch research prototype

The repository includes the original Roboflow dataset ZIPs in `data/uploads`, extracted images/COCO annotations in `data/datasets`, official model assets in `models`, and final research models/reports in `artifacts`. Large archives and model binaries use **Git LFS**. Install Git LFS before cloning or pulling:

```bash
git lfs install
git clone https://github.com/kg905/SeniorDesignProject.git
cd SeniorDesignProject
git lfs pull
```

Dataset attribution and declared licenses are retained in each export's `README.dataset.txt`, `README.roboflow.txt`, and COCO annotation metadata. Sources: [dat-1yih3/baby-z5nw7](https://universe.roboflow.com/dat-1yih3/baby-z5nw7) (declared Public Domain/CC0) and [dorsaf-0k2mw/baby-detection-zp6c1](https://universe.roboflow.com/dorsaf-0k2mw/baby-detection-zp6c1) (declared CC BY 4.0). Transfer chunks, virtual environments, credentials, inference checkpoints and superseded experiments remain local; they are not needed to reproduce the final run.

Pi 5 estimates posture and Pi 4 runs a portable alarm on the same private Wi-Fi network. Colab supports dataset review, feature extraction, training, and video-file testing. The repository began empty. The uploaded datasets have now been processed and two exploratory posture models trained: **neither meets the 95% target or is suitable for the baby alarm.** Read [DATASET_RESULTS.md](DATASET_RESULTS.md) for the actual results. Live monitoring rejects these exploratory artifacts.

**This is a senior-design research prototype, not a medical device or a demonstrated SIDS-prevention system.** It cannot measure breathing, oxygenation, airway clearance, or whether an infant is safe. Do not rely on it for infant supervision. Develop with dolls, recorded material obtained with permission, and responsible supervision; never place a baby in a dangerous position to create data.

Back, side, and prone are **posture labels**, not diagnoses. A visible face or raised head does not establish safety during sleep. Supervised awake tummy time is a different context that these twelve metrics cannot reliably recognize. This prototype therefore does not automatically suppress a prone alert when a face is visible. Follow established safe-sleep guidance, including the [AAP's guidance for parents](https://www.healthychildren.org/English/ages-stages/baby/sleep/Pages/How-to-Keep-Your-Sleeping-Baby-Safe-AAP-Policy-Explained.aspx). Development and clinical validation are different activities.

## Start in Colab

Open [notebooks/colab.ipynb](notebooks/colab.ipynb) in Google Colab. Download the supplied `cribwatch-source.zip` and upload it when the notebook asks. The notebook installs the package, verifies official model downloads, accepts exported dataset ZIPs, displays labeling widgets, trains a baseline, and analyzes an uploaded video. A CPU runtime is sufficient for this baseline. Persist your data and reports to a private Drive folder; Colab storage is temporary.

The source ZIP is a convenience snapshot for Colab, not a GitHub Release. You can also clone this repository with Git LFS to obtain the source, original dataset exports, model files and final research results. Publishing files to GitHub does not validate a model or publish the cloud environment.

## Local development

Python 3.11–3.13, 64-bit Linux. Raspberry Pi OS Bookworm (Python 3.11) or Trixie (Python 3.13) is the deployment target. Use the same package versions for training and inference. GPU acceleration is not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vision,notebook,test]'
cribwatch download-models
python -m pytest -q
```

Linux needs `libgles2`, `libegl1`, `libgl1`, and `libportaudio2` for the MediaPipe/OpenCV imports, even with CPU inference. On Debian/Ubuntu install OS packages using your normal package manager. In the provided cloud workspace use `bash deploy/setup-cloud.sh`, then `source deploy/activate-cloud.sh`. The cloud helper installs the missing GLES library under `/workspace/cribwatch-runtime` using signed Debian indexes and verified packages, with no privileged writes. Model files are downloaded over verified HTTPS from versioned official Google objects and checked against pinned SHA256 hashes.

## Dataset intake and labeling

The completed uploaded-data run reconstructed both exports from all 12 attachments, audited 5,814 images, extracted the requested features, trained metrics-only and image+metric baselines, and generated a review queue. The 4,814-image export already has posture labels; the 1,000-image export labels baby/face boxes only. Publisher labels remain distinct from independent human verification. Current results are in `artifacts/posture_baseline_v2`, `artifacts/image_research_final`, and `artifacts/dataset_audit_v2`.

To reproduce the exploratory workflow after extracting the original exports:

```bash
cribwatch audit-coco --dataset posture=/path/to/posture-export --dataset baby_face=/path/to/baby-face-export --output artifacts/new-audit
cribwatch extract artifacts/new-audit/images.csv --output artifacts/new-audit/features.csv --workers 2
cribwatch train artifacts/new-audit/features.csv --exploratory --output artifacts/new-metrics-run
cribwatch embed artifacts/new-audit/features.csv --output artifacts/new-audit/embeddings.npz
cribwatch train-image artifacts/new-audit/features.csv --embeddings artifacts/new-audit/embeddings.npz --output artifacts/new-image-run
cribwatch propose-unlabeled artifacts/new-audit/features.csv --embeddings artifacts/new-audit/embeddings.npz --classifier artifacts/new-image-run/posture.joblib --output artifacts/new-audit/review_queue.csv
```

This mode separates conservative source/duplicate groups, not individual babies. It quarantines contradictory labels on declared same-source images; near-duplicate frames showing posture changes are only grouped. Unknown-image suggestions never become ground truth. Standard training without `--exploratory` still requires reviewed labels and actual baby/session IDs. Do not invent IDs to bypass that requirement. Use `analyze-video --allow-exploratory` only for offline research; no corresponding bypass exists for live monitoring.

Requested sources:

- <https://universe.roboflow.com/dat-1yih3/baby-z5nw7>
- <https://universe.roboflow.com/dorsaf-0k2mw/baby-detection-zp6c1>

Initial direct Universe requests returned HTTP 403 and API requests required authentication. The user subsequently uploaded both COCO exports in verified chunks; their actual counts and annotation metadata were inspected locally. The exports declare Public Domain/CC0 and CC BY 4.0 respectively; underlying permissions were not independently audited. A baby bounding box is not a back/side/prone ground-truth label.

Export images through Roboflow using your authorized account, check the dataset license, and upload the ZIP to Colab. No API key is needed for this local-export route. Do not paste API keys into chat or commit them. No Roboflow SDK is required.

For uploads to this chat, the current execution transfer limit is **32 MiB per attached file**. Both original uploaded dataset ZIPs exceeded that limit and could not be inspected. Open [deploy/split-dataset.html](deploy/split-dataset.html) locally in a current browser, select an original dataset ZIP, prepare the parts, and download/upload all numbered ZIP parts plus the `.parts.json` manifest. Repeat for the other dataset. The helper runs entirely locally and uses 24 MiB payloads wrapped in valid ZIP files. It preserves the original images and annotations; do not rename or unzip the parts. On the receiving machine reconstruct them with:

```bash
cribwatch reassemble /path/to/dataset.zip.parts.json --parts-dir /path/to/parts \
  --output data/restored-dataset.zip
```

Reconstruction checks every part's size, ZIP CRC and SHA256, rejects incomplete sets and overwrites, and returns the complete archive's SHA256. The reconstructed ZIP can then enter the normal export-inspection pipeline. There is no need to publish infant images or create a public download link.

The HTML helper must be downloaded and opened outside the chat preview, which can restrict file saving. The updated helper uses ordinary download links retained until another dataset is prepared or the page closes; right-click **Save link as** is also available. Supported Chrome/Edge browsers additionally offer **Save all files to a folder**, which requests one folder permission and saves into a fresh subfolder. Canceled/blocked permissions produce visible guidance; downloads are never reported as completed merely because a link was clicked.

If this chat interface does not deliver the HTML file as a download, paste the contents of `deploy/split-in-colab.py` into a Colab code cell. It accepts original ZIP uploads and creates the same verifiable ZIP parts and manifest in a new `cribwatch_parts_...` folder. In Colab's Files sidebar, expand that folder and use each file's menu to download it individually. Upload those individual files here; downloading the entire folder as one archive would exceed the transfer limit again. The script's splitting/reconstruction logic was tested with a simulated Colab upload and a 25 MiB archive; hosted Colab UI execution is unverified.

```bash
cribwatch manifest /path/to/exported/images --output data/manifest.csv
```

This indexes images recursively, verifies image decoding, removes exact-byte duplicates, and leaves posture **unknown/unreviewed**. It does not guess labels from dataset titles or file paths. Original images are never moved. The manifest is the logical split; training later writes a split manifest rather than copying private images into extra folders.

In a notebook:

```python
from cribwatch.data import review_widget
review_widget('data/manifest.csv', batch_size=12)
```

For each image label posture as `back`, `side`, `prone`, or `unknown`; separately label face appearance as `visible`, `turned_away`, `occluded`, or `unknown`. These face labels are for review and future extension, not a trained airway/face-down classifier. Use `unknown` for ambiguous images, multiple people, insufficient view, or uncertain posture. Do not infer belly/back solely from a missing face. Visibility of the mouth/nose is not proof of an unobstructed airway.

Provide a **real baby ID** shared across every recording of that same child, and a session ID for a recording/capture session. These identifiers come from authorized source metadata or known recordings, not face recognition, posture classes, filenames, or dataset names. Public datasets without identity metadata cannot establish baby-independent performance. Adjacent frames and augmentations must keep the same baby ID. Exact hashes catch only exact copies: inspect near-duplicates and overlapping sources too.

For labeling assistance, review a seed with at least three usable examples of every posture, extract features, and generate proposals:

```bash
cribwatch extract data/manifest.csv --output data/seed_features.csv
cribwatch suggest data/seed_features.csv --output data/suggested.csv
```

Then `review_widget('data/suggested.csv')` groups proposed postures for visual review. Predictions do **not** become ground truth: all proposed labels remain unreviewed and posture stays unknown until a person reviews them. This reduces organization work but cannot remove the need for trustworthy labels. Make test-set annotations independently/blinded to suggestions before freezing the final evaluation; otherwise the test labels can inherit the model's mistakes. Review disagreement with a second reviewer. Once labels are complete, the reviewed features CSV can go directly to `train`.

For a fully reviewed image manifest:

```bash
cribwatch extract data/manifest.csv --output data/features.csv
cribwatch train data/features.csv --output artifacts/run1
```

## Metric definitions

All geometry uses **pixel coordinates of the processed image**. Feature schema 2 stretches RGB input to **640x640**, matching the supplied posture export; this can distort the source's physical aspect ratio and is not a calibration. The projected torso scale is the distance between shoulder midpoint and hip midpoint. Detection failures produce missing measurements and an unknown observation, not a reassuring zero. Keep camera orientation, crop and preprocessing consistent. Older schema-1 artifacts are rejected.

| Metric | Implementation and interpretation |
| --- | --- |
| `shoulder_vec_mag` | Shoulder width / projected torso length; dimensionless, not centimeters |
| `hip_vec_mag` | Hip width / projected torso length |
| `shoulder_angle_deg` | Undirected shoulder-line angle in image coordinates, in `[-90,90)` |
| `hip_angle_deg` | Same for hips; positive image y points down |
| `shoulder_hip_angle_diff` | Smallest axial angle difference, `0–90` degrees |
| Four `*_visibility` metrics | MediaPipe estimates for landmarks 11, 12, 23, 24; not calibrated infant probabilities |
| `face_detected` | One face landmark result in the frame; false means not detected, not necessarily face down |
| `face_landmark_count` | Usually 478 when detected; count is largely fixed by the model and is not a face-visibility percentage |
| `head_lift_ratio` | `(mid_shoulder_z - nose_z) / world_torso_length`, only with a detected face and visible pose nose; otherwise missing. **An estimated camera-depth proxy, not measured head height, lift, or airway clearance.** |

The model uses these 12 features in a class-balanced random forest. It learns multivariate posture boundaries from the reviewed training set. There are **no universal safe/unsafe ranges** supplied. Training writes per-class 5th/50th/95th feature percentiles for exploration; those are descriptive, not medical thresholds.

The heuristic landmark-quality threshold (0.5), minimum projected torso size (2% of image extent), model abstention score, and alarm delays are engineering choices. They are not clinically derived ranges. Pose landmarks are required in frame, with all four torso landmarks meeting visibility/presence thresholds. Multiple detected people/faces, poor landmarks, and low classifier scores yield `unknown`. Failure to detect a second person remains possible. Use a fixed single-crib view.

General-purpose MediaPipe models were not validated here on infants. These twelve features may be insufficient to distinguish supine from prone or recognize a face pressed into bedding. If held-out results are poor, collect suitable infant keypoint/posture annotations and evaluate a fine-tuned image/keypoint model, additional viewpoints, and camera calibration in a separate model-development iteration. Do not compensate by inventing reassuring cutoffs.

## Evaluation and the 95% goal

Standard training requires at least 15 identified babies, with at least two babies of each posture represented in each split. These are minimal software guardrails, **not a sufficient sample-size justification**. Approximately 60/20/20% of babies go to training/validation/test. No baby appears in multiple splits. Training fits usable poses; evaluation includes labeled pose failures as unknown errors. Validation selects an abstention threshold with a requested minimum coverage of 50%, prioritizing fewer side/prone images predicted as back, then fewer other accepted errors, then coverage. When the coverage constraint cannot be met, the report says so and chooses the greatest available coverage; it never treats all-abstaining output as a successful classifier. Random-forest scores are not calibrated probabilities. Exploratory mode uses source-image groups and cannot establish unseen-baby accuracy, even if its point estimates were high.

Outputs in a new run directory:

- `posture.joblib`: pipeline, feature schema, score threshold, training-input fingerprint.
- `split_manifest.csv`: every selected image's train/validation/test membership.
- `validation.json` and `test.json`: counts, confusion matrix, class recall, overall accuracy **counting abstentions as errors**, accepted-only accuracy, coverage, and unsafe-predicted-back count.
- `test_predictions.csv`: inspect individual errors, including pose failures.
- `training_feature_percentiles.json`: descriptive training distributions.

The report's 95% target flag requires overall accuracy and each class recall to reach 0.95 as **point estimates**. It is not certification or a SIDS/airway claim. Baby-cluster bootstrap intervals accompany accuracy; descriptive frame-binomial intervals accompany class recall and must not be treated as independent-frame evidence. More independent babies are needed for credible bounds. For example, even 59 independent successes out of 59 only gives a one-sided 95% exact binomial lower bound just above 95%; repeated frames from one child do not supply 59 independent cases. Plan sample size and confidence bounds before final testing.

Freeze a test cohort and preprocessing before tuning; after inspecting the final test results, do not tune and present the same test as independent validation. Also hold out cameras/rooms/recordings, check duplicate content, and stratify by lighting/IR, age, clothing, skin tone, blanket/crib occlusion, camera angles, and motion. Still-image performance does not establish continuous-monitoring performance. Use separately annotated continuous recordings to measure missed posture events, time to alarm, false alarms per hour, coverage, dropped frames, and Wi-Fi disconnect recovery. No video-event accuracy numbers are produced or claimed by this implementation.

Only load `joblib` models you trained or trust: its format can execute Python code. Models are sensitive research artifacts; keep infant data, metadata, video outputs, and reports private. `.gitignore` excludes generated data/models/artifacts. It does not replace proper access controls or consent.

## Video and alarm behavior

```bash
cribwatch analyze-video recording.mp4 --classifier artifacts/run1/posture.joblib \
  --output artifacts/recording.jsonl --fps 5
```

This processes video frames using **video time**, emits metrics/postures/state transitions, and never contacts the physical alarm. Outputs contain metadata only; raw video is not saved by the program. A file stream is unsuitable as a live camera watchdog; use `analyze-video` for Colab testing.

Live policy defaults:

- Continuous prone evidence for 1 second or side evidence for 2 seconds latches a `check_baby` alarm. Switching between side and prone does not erase accumulated concerning evidence.
- Continuous unknown evidence for 2 seconds triggers `monitor_fault`. An observation gap over 1.5 seconds immediately invalidates continuity and reports a fault. Missing data never establishes a back posture.
- A latched baby alert remains until stable back observations for 3 seconds **and** an explicit operator acknowledgement. Create the `--ack-file` on the Pi 5; an early acknowledgement is consumed without clearing the alert.
- The Pi 4 independently detects missing heartbeats after 5 seconds, including Pi 5 power loss or a hung inference process. Its startup state is a monitor fault until the sender connects. For the first observation seconds, `observing` is not an assertion of safety.

All delays require engineering and application-specific evaluation. The policy uses a monotonic clock and seconds, not frame counts. No automatic play/sleep inference or automatic face-visible alarm suppression is implemented.

```bash
# Pi 5, after actual training and hardware setup:
cribwatch monitor --picamera --classifier artifacts/run1/posture.joblib \
  --alarm-url http://192.168.50.1:8765 --ack-file /tmp/cribwatch-ack
# Explicit operator acknowledgement, after inspecting the situation:
touch /tmp/cribwatch-ack
```

For a USB camera omit `--picamera` and use `--source 0`. The live stream uses the latest captured frame, checks capture age, and rejects inference results older than 1.5 seconds. Heartbeat delivery runs on a separate thread with a processing-freshness watchdog. `--fps 5` is a request, not a guaranteed Pi frame rate. Measure actual latency/thermal behavior on your device. A solid-color image, camera obstruction, or changed viewpoint is not always detectable; this is another validation requirement.

See [deploy/raspberry-pi.md](deploy/raspberry-pi.md) for the private network, camera, GPIO, keys, services, and hardware checks.

## Verification status

See [VALIDATION.md](VALIDATION.md) for checks performed in the cloud and [DATASET_RESULTS.md](DATASET_RESULTS.md) for real uploaded-image metrics and exploratory posture results. Synthetic tests verify software behavior only. Independent unseen-baby accuracy, actual Pi hardware performance, continuous infant event detection and clinical outcomes remain unvalidated.
