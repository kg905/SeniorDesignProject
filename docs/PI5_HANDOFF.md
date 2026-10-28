# Pi 5 / Arducam: MediaPipe + 12 metrics + one classifier

Based on the supplied [SIDS_Pi5_Team_Handoff.pdf](reference/SIDS_Pi5_Team_Handoff.pdf), sections 2–4 and 7. The teammate reports Pi 5 4 GB, Debian 13 Trixie ARM64, Python 3.13.5, Arducam IMX290 CSI, working Picamera2, OpenCV 4.10.0 and sklearn 1.4.2 in `~/senior_design/sids_env`. These are reported hardware results, not tests performed here.

## Architecture and artifacts

**Keep MediaPipe and the original 12-feature schema.** One selected tree classifier maps metrics to back/side/prone: the original Random Forest or the validation-selected Random Forest/Extra Trees/gradient boosting recipe from [the improvement workflow](IMPROVE_MODEL.md). A state machine maps posture to risk and alarm. No MobileNet/image embeddings are used by this path. MediaPipe pose/face models extract features; they are not competing posture classifiers.

The PDF's V10 notebook and old receiver source were not supplied. We preserve actual repository feature/classifier/policy implementations, not unseen V10 logic. Do not fabricate `feature_ranges.joblib` or `risk_cutoffs.joblib`; current descriptive percentiles are JSON, and prototype alarm timing/category mapping is explicit code. These are not clinical safe ranges.

Build the ZIP **after review, extraction, training and evaluation**:

```text
SIDS_Pi5_Deployment/
  monitor_pi5.py / test_model.py / alarm_receiver.py
  requirements-pi5.txt
  README.md / PROTOCOL.md / SHA256SUMS.json
  cribwatch-camera.service / cribwatch-alarm.service
  models/
    posture_model.joblib
    test.json
    training_feature_percentiles.json
    pose_landmarker_lite.task   # or the selected Full/Heavy asset, exactly one
    face_landmarker.task
  src/cribwatch/...
```

Raw/private images, notebook runs and a second posture classifier are not bundled. Share permitted reference images and outputs separately, including unknown/error cases. Load only trusted team joblib files; pickle-based files can execute code.

## Python and MediaPipe compatibility

Do **not** install the downloaded MediaPipe 0.10.18 CPython 3.11 wheel into Python 3.13.

Official PyPI metadata for **MediaPipe 1.1.1** lists `mediapipe-1.1.1-py3-none-manylinux_2_28_aarch64.whl`, not a CPython-3.11-only build. This package/Tasks API works in the inspected Linux x86_64 Python 3.12 environment. ARM64 wheel availability is not proof of Pi imports, camera operation or performance; run the checks below on the actual device.

Use a separate environment with training's sklearn **1.7.2**, NumPy 2.2.6 and joblib 1.6.0. Do not load this artifact in the teammate's sklearn 1.4.2 environment. Keep `sids_env` intact. Retain OS Picamera2/libcamera through `--system-site-packages`; use one pip OpenCV contrib distribution matching training in the new environment, not two pip cv2 packages.

## Install on Pi 5

Unzip into `~/senior_design/SIDS_Pi5_Deployment`:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 libgles2 libegl1 libgl1 libportaudio2
python3 -m venv --system-site-packages ~/senior_design/vision_env
source ~/senior_design/vision_env/bin/activate
cd ~/senior_design/SIDS_Pi5_Deployment
python -m pip install -r requirements-pi5.txt
python -m pip check
python -c 'import platform, sys, cv2, numpy, sklearn, mediapipe; from picamera2 import Picamera2; print(platform.machine(), sys.version, cv2.__version__, numpy.__version__, sklearn.__version__, mediapipe.__version__, cv2.__file__)'
```

Confirm cv2 is the new environment's pinned contrib build. Do not pip-replace OS libcamera. For import/NumPy ABI failures, record errors and package paths rather than installing arbitrary wheels or bypassing verification.

Verify bundle hashes:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Bundle hashes match')
PY
```

Lite/face URLs and SHA256 are pre-pinned in `src/cribwatch/assets.py`. Full/Heavy variants are downloaded from versioned official HTTPS URLs with hashes recorded at download time. The selected asset hashes are bound into the trained classifier; the ZIP includes that exact pose asset plus face asset, and live/offline inference checks both. Do not rename a Full asset as Lite or substitute a different extractor after training. Full/Heavy speed on the actual Pi remains to be measured against the existing stale-frame limits.

## Offline check before camera startup

```bash
python test_model.py --model models --models models --showcase --output offline-check \
  /path/to/reference-back.jpg /path/to/reference-side.jpg /path/to/reference-prone.jpg
```

Writes annotated JPEGs and predictions.json with identical preprocessing, 12 metrics, pose box, scores and risk mapping. Compare the same images/model/asset hashes in Colab and Pi. Reference labels are truth, not a guarantee the prediction matches. Use new output folders.

## Live scan, box and risk level

```bash
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-risk moderate --log demo-recordings/run-001.jsonl --record demo-recordings/run-001.mp4
```

Run from the Pi desktop/HDMI for `--display`. For SSH/headless omit it and inspect logs/opt-in recordings. No remote video server is provided. Press s for annotated JPEG+JSON and q to stop (Ctrl-C headless). USB uses `--source 0` instead of `--picamera`. File input is not a time-faithful replay adapter; do not describe prerecorded input as live camera evidence.

Capture is 640x480 RGB888; little-endian libcamera BGR bytes are converted to RGB once. Verify color with a test chart. MediaPipe receives RGB stretched to 640x640, matching schema 2. This preserves current extraction, not physical geometry of already stretched dataset images.

Exact feature order:

```text
shoulder_vec_mag, hip_vec_mag, shoulder_angle_deg, hip_angle_deg,
shoulder_hip_angle_diff, left_shoulder_visibility, right_shoulder_visibility,
left_hip_visibility, right_hip_visibility, face_detected,
face_landmark_count, head_lift_ratio
```

Missing values are NaN internally/null in JSON. Imputation handles missing metrics, but invalid torso quality returns unknown. Multiple detections are unknown. Face absence does not prove face-down; head depth is not airway clearance.

The box is normalized xyxy from sufficiently visible landmarks with padding, mapped onto the exact original classified frame. It is a pose-derived region, not a separate baby detector, segmentation or airway box, and can miss occluded parts. Low-quality pose has no trusted box. An uncertain classification uses unknown risk even when a pose box is available.

Output includes posture, lower/moderate/high/unknown posture-risk category, raw model score, state, timestamp, reason, all 12 metrics, box/null, classifier hash and alarm state. Confidence is not medical risk or measured accuracy. No 95% badge is inferred from a high score.

## Alarm and Pi 4 integration

Use **TCP 5000** and the provided replacement receiver. The PDF's receiver code was not supplied; port equality does not establish wire compatibility. Provision the same protected ALARM_SHARED_KEY on both Pis using PROTOCOL.md. Add to the Pi 5 command:

```bash
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-host 192.168.50.1 --alarm-risk moderate --ack-file /tmp/cribwatch-ack
```

`moderate` triggers on side or prone; `high` triggers on prone only. Prototype dwell is side 2 seconds/prone 1 second, not medical thresholds. Unknown/stalled monitoring and link loss are separate faults. Alerts latch until explicit acknowledgement after back for 3 seconds. The Pi 4 button mutes 10 seconds without clearing status.

Pi 4 only needs Python and GPIO support:

```bash
sudo apt install python3 python3-gpiozero python3-lgpio
cd ~/senior_design/SIDS_Pi5_Deployment
python3 alarm_receiver.py
# Stop the console receiver before starting the GPIO listener:
python3 alarm_receiver.py --gpio 17 --mute-button 27
```

The shared key must already be securely loaded. Follow proper transistor-driven active-buzzer wiring; do not drive an arbitrary buzzer load directly from GPIO.

For headless automatic startup, adapt service templates to the actual username, paths, Pi 4 IP and protected `/etc/cribwatch.env`. Install only the corresponding service per Pi. Stop the camera service before foreground HDMI operation. Use systemctl start/stop/enable and journalctl checks. Services do not train/download models; only one process can own a camera or listening port.

## Hardware tests still required

Check actual color/orientation/box alignment, fixed-image parity with Colab, measured FPS/latency/temperature, trigger threshold/latch/ack/mute, camera/inference/Wi-Fi loss, reboot recovery, actual sound and battery life. The loop targets 5 Hz, not proven Pi throughput. Results older than 1.5 seconds are discarded; repeated display of a frame cannot accumulate new evidence. Receiver timeout is 5 seconds.

A powered-off Pi 4 or disconnected buzzer cannot report failure through that buzzer; independent hardware indication remains necessary. No actual Pi tests or improved reviewed-data accuracy were completed here. A live scan alone proves neither 95% accuracy nor SIDS prevention.
