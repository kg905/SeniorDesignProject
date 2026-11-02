"""Portable image + 12-metric posture inference; same display/alarm interface."""
import hashlib
import json
import math
from pathlib import Path

import torch

from .assets import verify_classifier_assets
from .features import FEATURES, LABELS, SCHEMA_VERSION, VISION_PREPROCESSING, observation_for_schema
from .fusion_model import ARCHITECTURE, MetricFusionNet, body_evidence, encode_metrics, image_tensor
from .vision import create_vision


class FusionPredictor:
    def __init__(self, model, *, models_dir="models", showcase=False):
        torch.set_num_threads(2)
        directory = Path(model) if Path(model).is_dir() else Path(model).parent
        config = json.loads((directory / "fusion.json").read_text())
        if (config.get("architecture") != ARCHITECTURE or config.get("features") != list(FEATURES)
                or config.get("preprocessing") != VISION_PREPROCESSING or config.get("schema_version") not in {2, 3}):
            raise ValueError("Incompatible fusion feature/architecture contract")
        weights = directory / "posture_fusion.pt"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        report = json.loads((directory / "test.json").read_text())
        if digest != config["model_sha256"] or digest != report.get("model_sha256"):
            raise ValueError("Fusion model/report checksum mismatch")
        if hashlib.sha256((directory / "fusion.json").read_bytes()).hexdigest() != report.get("config_sha256"):
            raise ValueError("Fusion feature/calibration configuration no longer matches its evaluation")
        if not showcase and not report.get("research_95pct_point_target_met"):
            raise ValueError("Fusion accuracy target not established; use --showcase for this experimental model")
        self.net = MetricFusionNet()
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.net.eval()
        self.config = config
        pose_name = verify_classifier_assets(config, models_dir)
        self.vision = create_vision(models_dir, pose_name=pose_name, adapter=config.get("vision_adapter"))
        self.metadata = {"model_sha256": digest, "feature_schema": config["schema_version"],
                         "overall_test_accuracy": report["accuracy_all_images_unknown_counts_as_error"],
                         "vision_adapter": config.get("vision_adapter"), "architecture": ARCHITECTURE}
        self.last_geometry = None

    def predict(self, rgb):
        observation = observation_for_schema(self.vision.observe(rgb), self.config["schema_version"])
        detection = getattr(self.vision, "last_detection_box", None)
        geometry = dict(observation.geometry or {})
        if detection is not None:
            geometry["body_box"] = detection
        geometry["box_source"] = "yolov8_body" if detection else "pose_landmarks" if observation.pose_box else "unavailable"
        self.last_geometry = geometry if observation.geometry is not None or detection is not None else None
        box = detection or observation.pose_box
        metrics = {k: float(v) if math.isfinite(v) else None for k, v in observation.metrics.items()}
        row = dict(observation.metrics, pose_ok=int(observation.pose_ok), quality_reason=observation.reason,
                   extraction_route=getattr(self.vision, "last_route", "full_frame"))
        if not body_evidence(row):
            return "unknown", 0., {}, observation.reason, metrics, box
        with torch.inference_mode():
            logits = self.net(image_tensor(rgb)[None], torch.from_numpy(encode_metrics(row, self.config["normalizer"]))[None])
            probabilities = torch.softmax(logits / self.config["temperature"], dim=1)[0].numpy()
        scores = dict(zip(LABELS, probabilities.astype(float).tolist()))
        score = max(scores.values())
        posture = max(scores, key=scores.get) if score >= self.config["score_threshold"] else "unknown"
        reason = "low_model_score" if posture == "unknown" else "ok" if observation.pose_ok else "image_with_missing_geometry"
        return posture, score, scores, reason, metrics, box

    def close(self):
        self.vision.close()
