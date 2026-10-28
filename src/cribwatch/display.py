"""A posture proxy and raw model score, never a medical risk probability."""
from datetime import datetime, timezone
import numpy as np

RISK = {"back": "lower", "side": "moderate", "prone": "high", "unknown": "unknown"}


def read_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def make_output(posture, confidence, reason, *, status=None, scores=None, inference_ms=None):
    status = status or {"state": "observing", "baby_alarm": False, "monitor_fault": posture == "unknown"}
    return {"schema_version": 1, "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "posture": posture, "risk": RISK[posture], "confidence": confidence, "reason": reason,
            "state": status["state"], "baby_alarm": bool(status["baby_alarm"]),
            "monitor_fault": bool(status["monitor_fault"]), "scores": scores or {},
            "inference_ms": inference_ms, "risk_basis": "experimental_posture_proxy_not_medical_risk"}


def draw_overlay(rgb, output, *, showcase=True, link="disabled"):
    import cv2
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    color = {"lower": (60, 190, 60), "moderate": (0, 190, 255),
             "high": (40, 40, 240), "unknown": (170, 170, 170)}[output["risk"]]
    box = output.get("bounding_box")
    if box is not None:
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = [int(v * scale) for v, scale in zip(box, (width, height, width, height))]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"Pose region | {output['risk'].upper()}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    lines = ["EXPERIMENTAL SHOWCASE - NOT A MEDICAL RISK ESTIMATE" if showcase else "RESEARCH POSTURE MONITOR",
             f"Posture: {output['posture']} | Posture risk: {output['risk']}",
             f"Model score: {output['confidence']:.1%} | State: {output['state']}",
             f"Reason: {output['reason']} | Alarm link: {link}", output["timestamp"]]
    banner = np.zeros((145, max(720, canvas.shape[1]), 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(banner, line, (10, 23 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    color if i == 1 else (235, 235, 235), 1, cv2.LINE_AA)
    frame = np.zeros((canvas.shape[0] + banner.shape[0], banner.shape[1], 3), np.uint8)
    frame[:banner.shape[0]] = banner
    frame[banner.shape[0]:, :canvas.shape[1]] = canvas
    return frame
