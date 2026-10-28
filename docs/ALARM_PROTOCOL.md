# MediaPipe / one-classifier alarm contract: TCP 5000, protocol version 1

Decision based on the attached handoff: use **TCP port 5000** because the existing Pi 4 prototype uses TCP. The new software does **not** use UDP 5005. Historical `cribwatch receiver` defaults to HTTP 8765 and remains for legacy baselines; do not mix those launch commands with this path.

**The old receiver's exact JSON/framing was not supplied. Replace it with `alarm_receiver.py` from this handoff on the Pi 4.** This is a defined updated interface, not a claim that an unseen TCP receiver accepts these messages.

## Framing and authentication

Each connection carries one UTF-8 JSON request terminated by LF (`\n`), and one LF-terminated JSON response. Maximum request is 8192 bytes. TCP reads can split messages arbitrarily; use newline framing, not a single unchecked `recv`.

The Pi 5 opens a challenge connection:

```json
{"op":"challenge"}
```

The Pi 4 returns `{"nonce":"<random one-use challenge>"}`. The Pi 5 then opens a status connection:

```json
{"op":"status","payload":{"schema_version":1,"timestamp":"2026-10-27T20:00:00.000+00:00","posture":"prone","risk":"high","confidence":0.92,"reason":"ok","state":"check_baby","baby_alarm":true,"monitor_fault":false,"scores":{"back":0.03,"side":0.05,"prone":0.92},"inference_ms":180.0,"risk_basis":"experimental_posture_proxy_not_medical_risk","experimental_showcase":true,"model_sha256":"<model hash>","nonce":"<challenge>"},"signature":"<HMAC hex>"}
```

The signature is HMAC-SHA256 over the payload serialized by Python `json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()`. The response is `{"accepted":true}` or false. The receiver validates the signature, state/boolean fields, fresh one-use nonce, and five-second challenge lifetime. A valid status is required before clearing the link fault. Wall clocks need not match; challenge/timeout accounting uses monotonic clocks.

Both devices must share the same random key of at least 32 bytes via `ALARM_SHARED_KEY`. Generate once and securely copy to the other device; two separately generated keys will not match. Do not commit keys. Authentication is not encryption; use the private WPA2/WPA3 LAN. No infant images are transmitted.

On one trusted Pi, create the protected file without printing the key:

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
```

Securely copy that same file to the other Pi's `~/.config/cribwatch/environment`, keeping mode 0600. Load it in each foreground test terminal without displaying it:

```bash
set -a
source ~/.config/cribwatch/environment
set +a
```

For the provided systemd templates, install the same file as root-owned `/etc/cribwatch.env` mode 0600; systemd loads it via EnvironmentFile. Never paste its value into source or chat. Private-LAN address/access-point instructions remain in the repository's `deploy/raspberry-pi.md`; configure networking locally so changes do not sever your only remote connection.

## Output semantics

| Field | Values/meaning |
| --- | --- |
| posture | `back`, `side`, `prone`, `unknown` |
| risk | `lower`, `moderate`, `high`, `unknown`; posture proxy only |
| confidence | Raw maximum classifier score, **not calibrated medical risk** |
| state | `observing`, `back_observed`, `check_baby`, `monitor_fault` |
| baby_alarm | Latched posture alert; do not derive it from a single raw score on Pi 4 |
| monitor_fault | Monitoring cannot currently be relied on; may coexist with baby_alarm |
| timestamp | UTC ISO8601; for logs, not replay security |
| reason | Includes `ok`, `low_model_score`, `no_pose`, `low_pose_quality`, multiple-detection/stale/inference fault reasons |

The classifier predicts back/side/prone from 12 MediaPipe metrics. No/poor pose, multiple detections, low confidence or stale input produce posture/risk unknown. Empty-crib tests are still required; MediaPipe pose presence is not a validated infant-presence detector. Runtime also logs `metrics`, normalized xyxy `bounding_box` (or null), and `alarm_trigger_risk`. The box comes from visible landmarks, not a separate object detector.

Status is sent about every 0.5 seconds by a separate sender thread. If processing stalls for three seconds, the sender marks monitoring faulty. If no valid status reaches Pi 4 for five seconds, Pi 4 raises a local link fault. A shutdown sends a best-effort explicit fault; timeout handles abrupt process/power loss on Pi 5.

`--alarm-risk moderate` triggers on sustained side or prone; `--alarm-risk high` triggers only on prone. Defaults are prone one second and side two seconds. Continuous above-trigger evidence accumulates across transitions. Below-trigger/unknown/back or an excessive observation gap resets that evidence. Latched alerts clear only with explicit acknowledgement after stable back for three seconds. These are prototype engineering defaults, not clinical cutoffs.

## Pi 4 launch and tests

Install Python and gpiozero/lgpio on Pi 4; no TensorFlow, NumPy/Pillow, sklearn, or MediaPipe is needed by the receiver. The launch script uses bundled source directly. GPIO 17 is an example BCM output for a **properly transistor-driven active buzzer**, not permission to drive an arbitrary load directly. Optional mute button is BCM 27 to ground with internal pull-up.

```bash
# Shared environment key must already be loaded securely.
python alarm_receiver.py                           # console-only first
python alarm_receiver.py --gpio 17 --mute-button 27 # hardware bench
```

Provide a stable private IP and restrict port 5000 to the intended LAN. Initial/no-link faults pulse the buzzer; a baby_alarm gives continuous sound. Mute lasts ten seconds and does not clear status. Test actual audible output, replay/wrong-key rejection, camera/process/Wi-Fi loss, reboot behavior, acknowledgement, and mute. A receiver without power cannot report its own power failure; independent hardware indication remains necessary.

Automated local TCP tests verify framing/authentication/replay/timeout behavior. They do not prove the team's existing receiver compatibility, private Wi-Fi configuration, GPIO wiring, or acoustic output. The new protocol must be adopted on both devices before an end-to-end hardware claim.
