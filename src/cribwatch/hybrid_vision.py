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
    if not required <= set(adapter) or set(adapter) - required - {"keypoint_layout", "strategy"}:
        raise ValueError("Incomplete hybrid adapter configuration")
    if adapter.get("keypoint_layout", "coco17") not in {"coco17", "torso4"}:
        raise ValueError("Choose explicit COCO-17 or left/right shoulder/hip torso-4 layout")
    if adapter.get("strategy", "fallback") not in {"fallback", "yolo_primary"}:
        raise ValueError("Unknown hybrid selection strategy")
    if adapter.get("strategy") == "yolo_primary" and adapter["kind"] != "yolo_mediapipe_keypoints_v1":
        raise ValueError("YOLO-primary selection requires keypoint mode")
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
        self.last_detection_box = None
        observation = self.mediapipe.observe(rgb)
        if observation.reason in {"multiple_people", "multiple_faces"}:
            return observation
        primary = self.adapter.get("strategy") == "yolo_primary"
        if observation.pose_ok and not primary:
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
        self.last_detection_box = np.clip(box / 640, 0, 1).tolist()
        padding = (box[2:] - box[:2]) * self.adapter["padding"]
        region = np.concatenate([np.maximum(box[:2] - padding, 0), np.minimum(box[2:] + padding, 640)]) / 640
        if primary:
            candidate = self._keypoint_fallback(rgb, region, results[0], unknown("no_yolo_keypoints"))
            if candidate.pose_ok:
                # Retain a real MP depth proxy only when both see one face and
                # the MP torso is contained in this same detected body region.
                depth = observation.metrics.get("head_lift_ratio", math.nan)
                anchors = (observation.geometry or {}).get("points", {})
                torso = [anchors.get(name) for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
                associated = all(p is not None and region[0] <= p["x"] <= region[2] and region[1] <= p["y"] <= region[3] for p in torso)
                if observation.pose_ok and associated and math.isfinite(depth) and observation.metrics.get("face_detected") == candidate.metrics.get("face_detected") == 1:
                    from dataclasses import replace
                    candidate = replace(candidate, metrics=dict(candidate.metrics, head_lift_ratio=depth),
                                        geometry=dict(candidate.geometry, depth_source="associated_mediapipe_world_pose"))
                return candidate
            if observation.pose_ok:
                self.last_route = "full_frame"
                return observation
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
        indices = ((11, 12, 23, 24) if self.adapter.get("keypoint_layout") == "torso4"
                   else (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28))
        if xy.shape != (1, len(indices), 2) or confidence.shape != (1, len(indices)):
            return unknown("unsupported_yolo_keypoints")
        # COCO-17 -> MediaPipe indices. Only image x/y and native keypoint
        # confidence are available; no world z is invented from 2D geometry.
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
                               face_landmark_count=landmark_count, source="yolov8_keypoints")

    def close(self):
        if self.mediapipe is not None:
            self.mediapipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
