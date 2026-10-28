# Pi 5 vision / Pi 4 portable alarm

**Legacy baseline instructions.** For the MediaPipe + one-classifier camera overlay and TCP 5000 receiver selected after the team's hardware handoff, use [docs/PI5_HANDOFF.md](../docs/PI5_HANDOFF.md), [docs/ALARM_PROTOCOL.md](../docs/ALARM_PROTOCOL.md), and [docs/TRAIN_AND_SHOWCASE.md](../docs/TRAIN_AND_SHOWCASE.md). Do not mix these HTTP/MediaPipe service commands with the new receiver. The protected shared-key and private-LAN procedures below remain applicable.

These instructions target **64-bit Raspberry Pi OS Bookworm or Trixie**, Python 3.11–3.13, and a single infant in a fixed crib view. They have not been executed on physical Pi hardware here. A 32-bit OS is unsupported. The chosen MediaPipe release provides an ARM64 wheel; wheel availability alone does not establish device performance. Prefer a cooled Pi 5 for inference.

The current uploaded-data artifacts are exploratory and below the accuracy target; **do not enable them for live baby monitoring**. The software rejects them in the live monitor. They can be used for permitted offline-video research with `analyze-video --allow-exploratory`. Hardware instructions below describe the development setup for a future independently evaluated model, not approval to deploy the current research artifacts.

## Install before switching to an isolated Wi-Fi network

Copy this project to `/home/pi/SeniorDesignProject` on both devices, or adapt the paths/user in the service files. Use the real account name if it is not `pi`.

On Pi 5:

```bash
sudo apt update
sudo apt install python3-venv python3-picamera2 libgles2 libegl1 libgl1 libportaudio2
cd /home/pi/SeniorDesignProject
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[vision]'
cribwatch download-models
```

`--system-site-packages` exposes the OS Picamera2/libcamera bindings; do not replace those with an arbitrary pip libcamera package. Check `rpicam-hello --list-cameras` (or the OS-provided equivalent). Stop other camera programs before the monitor. The Picamera2 adapter requests `RGB888`, whose byte order on the little-endian Pi is converted from BGR to RGB for MediaPipe. Verify color with a test chart on actual hardware. USB/V4L2 cameras use `--source 0`. Do not run both adapters simultaneously against one camera.

Copy your **own trained** `artifacts/run1/posture.joblib` from Colab to the corresponding Pi 5 path. Do not substitute a synthetic-test model. Confirm all package versions match the training environment (`python -m pip check`); serialized scikit-learn models are not cross-version stable.

On Pi 4:

```bash
sudo apt update
sudo apt install python3-venv python3-lgpio
cd /home/pi/SeniorDesignProject
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[gpio]'
```

The receiver needs no vision models, camera, or GPU. Wire a 3.3V-compatible, **transistor-driven active buzzer** to the selected BCM output (example 17), using an appropriate driver and shared ground; do not drive an arbitrary buzzer load directly from a GPIO pin. An optional normally-open mute button connects BCM 27 to ground using the internal pull-up. Verify voltage/current ratings and wiring independently. The software does not verify that a buzzer physically sounds or that its battery is charged.

## Shared key (one value generated once, installed securely on both)

Generate a random key locally and store it in `/etc/cribwatch.env` on both devices. Keep the file accessible only to the service administrator. Never commit it, echo it in logs, or send it through chat. The file contains `ALARM_SHARED_KEY=<random value of at least 32 bytes>`. Generate a 32-byte random value with Python's `secrets.token_hex(32)` and write it directly to a protected file rather than pasting it into a tracked config. Both Pis need the **same** key; two independently generated keys will not work.

For example, on one trusted device:

```bash
sudo python3 - <<'PY'
import os, secrets
fd = os.open('/etc/cribwatch.env', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    f.write('ALARM_SHARED_KEY=' + secrets.token_hex(32) + '\n')
PY
```

Copy that file using a secure administrator-controlled channel to the second Pi, keeping its permissions at 0600. For a foreground test, an administrator can export the file's variable in the same terminal without displaying the key. The service manager loads it directly for normal startup. There are no embedded/default deployment keys.

Each message uses HMAC-SHA256 plus a fresh, one-use receiver nonce expiring after five seconds. Old signed messages cannot simply be replayed and no wall-clock/NTP agreement is needed. LAN HTTP is authenticated but **not encrypted**. Use an isolated WPA2/WPA3 network and a firewall; use a TLS endpoint/VPN for an untrusted network. This is not protection against a compromised Pi or denial-of-service attack. No images are transmitted to the portable alarm.

## Private Wi-Fi

The simplest arrangement is Pi 4 as a Wi-Fi access point and Pi 5 as a client. Another Wi-Fi chip/access point can provide the LAN if both devices can reach each other. Internet access is unnecessary at runtime after dependencies, model files, and classifier are installed. Colab does not need to reach the Pi LAN; test it with uploaded video instead.

Configure network changes from a local console so you do not disconnect your only remote session. On current Raspberry Pi OS using NetworkManager:

```bash
# Pi 4: --ask prompts for the AP's strong WPA password.
sudo nmcli --ask device wifi hotspot ifname wlan0 con-name cribwatch-ap ssid CribWatch
sudo nmcli connection modify cribwatch-ap ipv4.addresses 192.168.50.1/24 ipv4.method shared
sudo nmcli connection up cribwatch-ap
# Pi 5: join the same LAN; --ask avoids putting a password in shell history.
sudo nmcli --ask device wifi connect CribWatch
```

Verify the actual interface names first with `nmcli device status`. Assign Pi 4's stable address to the monitor's `--alarm-url`. Restrict port 8765 to the Pi 5/private subnet using your firewall. Do not expose this receiver to the Internet. A single Pi radio usually cannot remain a client of your household Wi-Fi while also serving this AP; install dependencies first or use Ethernet/a second radio for maintenance.

## Run and supervise

Pi 4 (environment variable already loaded securely into this shell):

```bash
source /home/pi/SeniorDesignProject/.venv/bin/activate
cribwatch receiver --host 0.0.0.0 --port 8765 --gpio 17 --mute-button 27
```

Omit GPIO arguments for a console-only bench test. A baby alarm is a continuous tone; a monitor/link fault is a pulsed tone. The mute button temporarily silences either for ten seconds without clearing the underlying alert. The hardware needs a separate independent power/failure indicator if you need to detect receiver crashes or receiver power loss: this software cannot sound a buzzer after the Pi 4 loses power.

Pi 5:

```bash
source /home/pi/SeniorDesignProject/.venv/bin/activate
cd /home/pi/SeniorDesignProject
cribwatch monitor --picamera --models models --classifier artifacts/run1/posture.joblib \
  --alarm-url http://192.168.50.1:8765 --ack-file /tmp/cribwatch-ack
```

For service startup, review the supplied `.service` templates, replace `pi` and paths if necessary, add the appropriate camera/GPIO group permissions, and install only on the corresponding device:

```bash
sudo cp deploy/cribwatch-receiver.service /etc/systemd/system/  # Pi 4 only
# OR deploy/cribwatch-vision.service on Pi 5
sudo systemctl daemon-reload
sudo systemctl enable --now cribwatch-receiver
sudo journalctl -u cribwatch-receiver -f
```

The Pi 5 equivalent service is `cribwatch-vision`. Models/dependencies must be installed before enabling it; the service does not train or download anything. A noninteractive service cannot ask for keys. Its first useful readiness checks are JSON state output on Pi 5, accepted heartbeats at Pi 4, and the actual buzzer response. Merely having a running PID is insufficient.

## Hardware acceptance checks still required

Use bench inputs and permitted recordings, not unsafe positioning of an infant:

1. Confirm camera image orientation, coverage, color, and timestamp freshness.
2. Verify back/side/prone predictions against independently labeled recordings on the Pi; measure inference latency, temperature, CPU load, and sustained sample rate.
3. Verify temporal dwell, latched alerts, explicit acknowledgement after stable back, and ten-second mute behavior.
4. Cover/unplug the camera; stop/hang the vision process; disconnect Wi-Fi; power off Pi 5. Confirm distinct monitor-fault behavior and detection time.
5. Reboot each device separately and together; verify startup faults, service recovery, and no silent stale status.
6. Verify weak/wrong keys are rejected and captured messages cannot be replayed. Verify firewall boundaries and actual audible output/battery duration.
7. Test Pi 4 power loss and buzzer disconnection using the separate hardware failure indicator. They are not detectable by a powered-off receiver.

Do not claim the system meets a performance target until these checks and a properly designed infant-posture evaluation are completed.
