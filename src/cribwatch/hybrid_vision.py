"""Explicit alternatives: MediaPipe crop recovery, optionally YOLO 2D fallback."""

import hashlib
import math
from pathlib import Path

import numpy as np

from .assets import YOLO_ASSETS
from .features import extract_metrics, unknown
from .vision import Vision


def validate_adapter(adapter):
    if not isinstance(adapter, dict) or adapter.get("kind") not in {"yolo_mediapipe_roi_v1", "yolo_mediapipe_keypoints_v1"}:
        raise ValueError("Unknown vision adapter")
    required = {"kind", "yolo_name", "yolo_sha256", "confidence", "padding"}
    if set(adapter) != required:
        raise ValueError("Incomplete hybrid adapter configuration")
    name, digest = adapter["yolo_name"], adapter["yolo_sha256"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("YOLO weights must have a local .pt basename")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid YOLO asset checksum")
    if name in YOLO_ASSETS and digest != YOLO_ASSETS[name][1]:
        raise ValueError("Pinned YOLO asset checksum mismatch")
    for key, lo, hi in (("confidence", .01, 1.), ("padding", 0., 1.)):
        if not isinstance(adapter[key], (int, float)) or not math.isfinite(adapter[key]) or not lo <= adapter[key] <= hi:
            raise ValueError(f"Invalid hybrid {key}")
    return adapter


class HybridVision:
    def __init__(self, model_dir, *, pose_name="pose_landmarker_lite.task", adapter):
        self.adapter = validate_adapter(adapter)
        self.mediapipe = None
        path = Path(model_dir) / adapter["yolo_name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != adapter["yolo_sha256"]:
            raise ValueError("YOLO asset checksum mismatch")
        # Lazy import: baseline inference and the Pi 4 receiver do not need Torch.
        import torch
        torch.set_num_threads(2)
        from ultralytics import YOLO
        self.yolo = YOLO(str(path))
        if self.yolo.task != "pose":
            raise ValueError("Use a trusted YOLO pose model, not a posture classifier")
        self.mediapipe = Vision(model_dir, pose_name=pose_name)
        self.last_route = "not_started"

    def observe(self, rgb):
        rgb = Vision._prepare(rgb)
        self.last_route = "full_frame"
        observation = self.mediapipe.observe(rgb)
        if observation.pose_ok or observation.reason in {"multiple_people", "multiple_faces"}:
            return observation
        self.last_route = "unresolved"
        # Ultralytics numpy input is BGR; MediaPipe and public API input are RGB.
        results = self.yolo.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]),
            imgsz=640, conf=self.adapter["confidence"], device="cpu", verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes is not None else np.empty((0, 4))
        if len(boxes) > 1:
            return unknown("multiple_people")
        if len(boxes) != 1:
            return observation
        box = np.asarray(boxes[0], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
            return unknown("invalid_body_region")
        padding = (box[2:] - box[:2]) * self.adapter["padding"]
        region = np.concatenate([np.maximum(box[:2] - padding, 0), np.minimum(box[2:] + padding, 640)]) / 640
        self.last_route = "yolo_crop"
        cropped = self.mediapipe.observe_region(rgb, region)
        if (cropped.pose_ok or cropped.reason in {"multiple_people", "multiple_faces"}
                or self.adapter["kind"] == "yolo_mediapipe_roi_v1"):
            return cropped
        return self._keypoint_fallback(rgb, region, results[0], cropped)

    def _keypoint_fallback(self, rgb, region, result, failed):
        from types import SimpleNamespace
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.conf is None:
            return failed
        xy = keypoints.xy.cpu().numpy()
        confidence = keypoints.conf.cpu().numpy()
        if xy.shape != (1, 17, 2) or confidence.shape != (1, 17):
            return unknown("unsupported_yolo_keypoints")
        # COCO-17 -> MediaPipe indices. Only image x/y and native keypoint
        # confidence are available; no world z is invented from 2D geometry.
        indices = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
        points = [SimpleNamespace(x=0., y=0., z=0., visibility=0., presence=0.) for _ in range(33)]
        for source, target in enumerate(indices):
            x, y = xy[0, source] / 640
            score = float(confidence[0, source])
            points[target] = SimpleNamespace(x=float(x), y=float(y), z=0., visibility=score, presence=score)
        bounds = np.array(region) * 640
        x1, y1 = np.maximum(0, np.floor(bounds[:2])).astype(int)
        x2, y2 = np.minimum(640, np.ceil(bounds[2:])).astype(int)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return failed
        face_count, landmark_count = self.mediapipe.face_summary(rgb[y1:y2, x1:x2])
        if face_count > 1:
            return unknown("multiple_faces")
        self.last_route = "yolo_keypoints"
        return extract_metrics(points, 640, 640, world=None, face_count=face_count,
                               face_landmark_count=landmark_count)

    def close(self):
        if self.mediapipe is not None:
            self.mediapipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
