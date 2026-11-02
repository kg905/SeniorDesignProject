"""Capture manually labeled camera frames with genuine session metadata."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from uuid import uuid4

from .camera import LatestCamera
from .frame_roi import crop_frame, parse_roi


def capture(output, *, posture, subject_id, session_id, subject_kind="doll", seconds=30,
            fps=1., picamera=False, source="0", roi=None, preview=False):
    import cv2
    from PIL import Image
    if posture not in {"back", "side", "prone", "unknown", "no_baby"}:
        raise ValueError("Choose a real manually observed posture label")
    if subject_kind not in {"doll", "infant"}:
        raise ValueError("Subject kind must be doll or infant")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value) for value in (subject_id, session_id)):
        raise ValueError("Use simple actual subject/session IDs (letters, digits, underscore, hyphen)")
    if not 0 < seconds <= 600 or not 0 < fps <= 10:
        raise ValueError("Choose 0–600 seconds and 0–10 frames/second")
    root = Path(output)
    clip = uuid4().hex[:12]
    directory = root / posture / session_id / clip
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "capture.json").write_text(json.dumps({"subject_id": subject_id, "subject_kind": subject_kind,
        "session_id": session_id, "clip_id": clip, "label": posture, "camera_roi": roi,
        "started_utc": datetime.now(timezone.utc).isoformat(), "label_source": "operator_supplied",
        "instruction": "Keep the specified posture throughout this capture; review before training."}, indent=2))
    columns = ["relative_path", "baby_id", "session_id", "subject_id", "subject_kind"]
    metadata = root / "metadata.csv"
    if metadata.exists():
        with metadata.open(newline="") as f:
            if next(csv.reader(f), None) != columns:
                raise ValueError("Existing metadata has a different format; use a separate capture root")
    camera, saved, seen = LatestCamera(source, picamera=picamera), 0, -1
    start, next_capture = time.monotonic(), time.monotonic()
    try:
        while time.monotonic()-start < seconds:
            frame = camera.get()
            now = time.monotonic()
            if frame is None or frame[0] == seen or now-frame[1] > 1.:
                time.sleep(.02)
                continue
            seen = frame[0]
            rgb = crop_frame(frame[2], roi)
            if now >= next_capture:
                path = directory / f"{saved:05d}-{time.time_ns()}.jpg"
                Image.fromarray(rgb).save(path, quality=95)
                first = not metadata.exists()
                with metadata.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    if first:
                        writer.writeheader()
                    writer.writerow({"relative_path": path.relative_to(root).as_posix(),
                        "baby_id": subject_id if subject_kind == "infant" else "", "session_id": session_id,
                        "subject_id": subject_id, "subject_kind": subject_kind})
                saved += 1
                next_capture = now + 1/fps
                print(f"Saved {posture} frame {saved}; subject {subject_id}, session {session_id}", flush=True)
            if preview:
                cv2.imshow("Capture: operator label " + posture, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xff == ord("q"):
                    break
    finally:
        camera.close()
        if preview:
            cv2.destroyAllWindows()
    if not saved:
        raise RuntimeError("No fresh camera frames captured; check camera configuration")
    print("Review the labels, then ZIP the entire capture root:", root, flush=True)
    return saved


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--posture", required=True, choices=["back", "side", "prone", "unknown", "no_baby"])
    p.add_argument("--subject-id", required=True)
    p.add_argument("--session-id", required=True)
    p.add_argument("--subject-kind", choices=["doll", "infant"], default="doll")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--fps", type=float, default=1.)
    p.add_argument("--picamera", action="store_true")
    p.add_argument("--source", default="0")
    p.add_argument("--roi", type=parse_roi)
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    capture(a.output, posture=a.posture, subject_id=a.subject_id, session_id=a.session_id,
            subject_kind=a.subject_kind, seconds=a.seconds, fps=a.fps, picamera=a.picamera,
            source=a.source, roi=a.roi, preview=a.preview)
