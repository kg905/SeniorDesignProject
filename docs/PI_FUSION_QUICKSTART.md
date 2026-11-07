# Fusion model: Colab, Pi 5 camera and Pi 4 alarm

This guide matches the [uploaded fusion package, model `98dbc3ec`](../artifacts/team_test_runs/fusion-49p2-98dbc3ec/README.md). Keep its archive unchanged in GitHub and unzip it into a new folder on the devices. Future Colab exports use the same workflow with their own matching models and assets.

## What goes where

```text
BabyTraining-V4.zip → Colab training → SIDS_Pi5_Fusion_Geometry.zip
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          Pi 5 + overhead Arducam                           Pi 4 + buzzer/button
          MediaPipe + YOLO recovery                          alarm_receiver.py
          12 metrics + image/metric network                 no ML dependencies
          posture → risk → timed alarm policy
                     │                                               ▲
                     └──── authenticated status over TCP 5000 ────────┘
```

Pi 5 runs all vision and classification. Pi 4 receives the resulting alarm/fault status on the private Wi-Fi network and operates the buzzer. Camera images and the training dataset are not sent to Pi 4. Colab is needed for training/export, not during live monitoring.

## 1. Colab and the two ZIPs

Open [fusion_geometry_colab.ipynb](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/fusion_geometry_colab.ipynb). Section 3 accepts **BabyTraining-V4.zip** from your computer, or from the `ZIP_IN_DRIVE` path. Its `Baby Training - V1` enclosing folder is supported. Sections 4–6 import, extract and train. New **section 6A** reports accuracy, macro/weighted F1, precision, recall and prediction coverage, including both full-test and assigned-only evaluations. Section 8 exports **SIDS_Pi5_Fusion_Geometry.zip**.

The already uploaded [deployment ZIP](../artifacts/team_test_runs/fusion-49p2-98dbc3ec/SIDS_Pi5_Fusion_Geometry.zip) is ready for team testing without retraining. Download it through GitHub's **Download raw**, not the small Git LFS pointer. Its SHA256 is `5eb641aa59751c38fdbf031204e82d16cb2d00c5d1fbeff02ff24d3f03d964b3`. Its report is 49.2% overall accuracy on 512 test images; 95% is not established.

The dataset attachment could not be transferred into this environment because it exceeds 32 MiB. It is being provided through Drive; the repository currently contains its [layout and usage instructions](../data/team_datasets/README.md), not that ZIP. Neither Pi needs the raw dataset.

## 2. Unpack on both Pis

Copy the deployment ZIP onto each Pi, for example into `~/Downloads`. On each device:

```bash
mkdir -p ~/senior_design/CribWatch_Fusion_98dbc3ec
unzip -n ~/Downloads/SIDS_Pi5_Fusion_Geometry.zip -d ~/senior_design/CribWatch_Fusion_98dbc3ec
cd ~/senior_design/CribWatch_Fusion_98dbc3ec
python3 - <<'PY'
import hashlib, json
from pathlib import Path
for name, expected in json.loads(Path('SHA256SUMS.json').read_text()).items():
    digest = hashlib.sha256()
    with Path(name).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == expected, name
print('All bundled file hashes match')
PY
```

Use a fresh folder; do not merge this package with an older deployment. It includes `models/posture_fusion.pt`, `models/fusion.json`, `models/test.json` and the matching detector assets. The final posture model is a neural network; it does not use the legacy `posture_model.joblib` path described in parts of the bundled README.

## 3. Pi 5: install the vision environment

These commands target the previously reported 64-bit Pi OS / Debian setup with working Picamera2. Keep the existing working camera environment intact and use this separate environment:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 libgles2 libegl1 libgl1 libportaudio2
python3 -m venv --system-site-packages ~/senior_design/fusion_env
source ~/senior_design/fusion_env/bin/activate
cd ~/senior_design/CribWatch_Fusion_98dbc3ec
python -m pip install -r requirements-pi5.txt
python -m pip install -r requirements-hybrid.txt
python -m pip install --no-deps ultralytics==8.3.200
python -c 'import torch, torchvision, cv2, mediapipe, ultralytics; from picamera2 import Picamera2; print("Vision imports OK", torch.__version__, torchvision.__version__, cv2.__version__)'
```

Both requirement files are needed for fusion. Preserve OS Picamera2/libcamera with `--system-site-packages`; do not install an unrelated wheel over libcamera. The pinned pip OpenCV package is `opencv-contrib-python`; installing `opencv-python` as well creates competing `cv2` installations. Ultralytics is installed with `--no-deps` for that reason. Its package metadata names `opencv-python`, so `pip check` may report that dependency even though the contrib build supplies `cv2`; confirm imports rather than adding a second build blindly.

If installation or imports fail, keep the exact error and Python/OS architecture details. These packages were checked in the development environment; actual Pi performance and camera operation still need verification.

## 4. Pi 4: prepare the receiver and private network

Put both Pis on the existing private Wi-Fi network. The examples below use **192.168.50.1 as the Pi 4 address**; replace it with the actual Pi 4 address and change the username `pi` if needed. This is not an instruction to reconfigure your current network.

On Pi 4:

```bash
sudo apt update
sudo apt install python3 python3-gpiozero python3-lgpio
mkdir -p ~/.config/cribwatch
chmod 700 ~/.config/cribwatch
hostname -I
```

Allow TCP 5000 between these devices on the private LAN if a firewall is enabled. Use `alarm_receiver.py` from this same package; an older receiver on port 5000 may use a different protocol. Pi 4 needs no Torch, MediaPipe, YOLO or training environment.

## 5. Provision the same authentication key on both Pis

If the existing pair already has a working protected `ALARM_SHARED_KEY`, reuse it. Otherwise generate one file on Pi 5, once. This command refuses to overwrite an existing key:

```bash
python3 - <<'PY'
import os, secrets
from pathlib import Path
folder = Path.home() / '.config/cribwatch'
folder.mkdir(parents=True, exist_ok=True)
fd = os.open(folder / 'environment', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('ALARM_SHARED_KEY=' + secrets.token_hex(32) + '\n')
PY
scp ~/.config/cribwatch/environment pi@192.168.50.1:~/.config/cribwatch/environment
```

Copy the same file; generating different keys on each device will prevent communication. Keep the key out of GitHub and screenshots. In the foreground test terminal on **each Pi**:

```bash
chmod 600 ~/.config/cribwatch/environment
set -a
source ~/.config/cribwatch/environment
set +a
```

The [alarm protocol](ALARM_PROTOCOL.md) explains challenge/HMAC authentication. It authenticates status messages; the private Wi-Fi supplies network protection.

## 6. Start Pi 4, then Pi 5

First, on Pi 4, start a console-only receiver to check the connection:

```bash
cd ~/senior_design/CribWatch_Fusion_98dbc3ec
python3 alarm_receiver.py
```

On Pi 5, with its key loaded in the same terminal:

```bash
source ~/senior_design/fusion_env/bin/activate
cd ~/senior_design/CribWatch_Fusion_98dbc3ec
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-host 192.168.50.1 --alarm-port 5000 --alarm-risk moderate \
  --ack-file /tmp/cribwatch-ack \
  --log demo-recordings/run-001.jsonl --record demo-recordings/run-001.mp4
```

Use `--display` from the Pi desktop/HDMI; omit it for headless SSH. Use new log/recording names for each run. The `--showcase` option is needed because this evaluated research model has not met the 95% target. Startup can show a monitoring fault until fresh inference and the authenticated link are working. Inspect Pi 4's status output and Pi 5's `alarm_link` field.

For an offline check on Pi 5, use your own reference images:

```bash
python test_model.py --model models --models models --showcase --output offline-check-001 \
  /path/to/back.jpg /path/to/side.jpg /path/to/prone.jpg
```

Still-photo previews show classification; they do not exercise the live alarm timing or activate the Pi 4 alarm.

After the console receiver is working, stop it with Ctrl-C. On Pi 4, start the GPIO receiver:

```bash
export GPIOZERO_PIN_FACTORY=lgpio
python3 alarm_receiver.py --gpio 17 --mute-button 27
```

BCM GPIO 17 is the example output; BCM 27 is the optional mute button to ground using the internal pull-up. Match your actual wiring. Drive the active buzzer through a suitable transistor/driver rather than connecting an arbitrary buzzer load directly to a GPIO. Only one receiver can listen on port 5000 at a time.

## 7. Risk, alarm and acknowledgement

| Observation | Default behavior |
| --- | --- |
| Back | Lower posture-risk category |
| Side | Moderate posture-risk category; trigger delay 2 seconds |
| Prone | High posture-risk category; trigger delay 1 second |
| Unknown | Unknown risk; monitoring fault after 2 seconds |
| Stale inference / excessive observation gap | Monitoring fault; stale frames do not supply new evidence |
| No authenticated status for 5 seconds on Pi 4 | Pi 4 link fault |

`--alarm-risk moderate` enables side and prone alerts; `--alarm-risk high` enables prone alerts only. Continuous above-trigger observations accumulate evidence; back/unknown or an excessive gap resets it. These timings are engineering defaults. Risk is assigned from predicted posture, not measured breathing or airway blockage, and a visible face does not automatically downgrade prone.

Posture alarms latch. Once the system has fresh back observations for at least 3 seconds, acknowledge on Pi 5:

```bash
touch /tmp/cribwatch-ack
```

The Pi 4 mute button silences output for 10 seconds without clearing the alarm. A latched posture alert gives continuous sound; monitoring/link faults pulse the buzzer. Test both conditions on the actual device.

## 8. Repeatable team checks and automatic startup

Test labeled doll back/side/prone sequences, transitions, unknown/empty-crib scenes, camera loss, blocked/stalled inference, Wi-Fi loss, mute, acknowledgement and reboot. Save the corresponding JSONL and MP4 under distinct run names. Confirm real latency, frame rate and audible output. Software checks here do not establish those physical results.

For startup services, use the archive's `cribwatch-camera.service` **only on Pi 5** and `cribwatch-alarm.service` **only on Pi 4**. Edit `User`, `WorkingDirectory` and `ExecStart` for the actual accounts, new folder and Pi 5 `fusion_env` path; set the correct Pi 4 address. Install the matching protected key file at `/etc/cribwatch.env` mode 0600 as expected by the templates. Then install/enable each service on its device. Stop its service before a foreground camera or receiver test so two processes do not compete for the hardware or port.

Older ZIPs and their measured reports remain preserved. Do not swap a detector, feature implementation or posture weight file independently of the matching model configuration.
