"""MediaPipe -> exactly 12 features -> one posture classifier, shared offline/live."""
import hashlib
import json
import math
from pathlib import Path

from .learning import Predictor
from .assets import verify_classifier_assets
from .vision import Vision


class MetricsPredictor:
    def __init__(self, model, *, models_dir="models", showcase=False):
        path = Path(model)
        if path.is_dir():
            path = path / "posture_model.joblib"
        self.predictor = Predictor(path, allow_exploratory=showcase)
        if self.predictor.artifact.get("input_kind", "metrics") != "metrics":
            raise ValueError("Use the 12-metric classifier, not the old image model")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report_path = path.parent / "test.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if not showcase and (report.get("model_sha256") != digest or not report.get("research_95pct_point_target_met")):
            raise ValueError("95% target not established for this artifact; use --showcase for an experimental demo")
        adapter = self.predictor.artifact.get("vision_adapter")
        self.metadata = {"model_sha256": digest, "feature_schema": self.predictor.artifact["schema_version"],
                         "vision_adapter": adapter}
        pose_name = verify_classifier_assets(self.predictor.artifact, models_dir)
        if adapter is None:
            self.vision = Vision(models_dir, pose_name=pose_name)
        else:
            from .hybrid_vision import HybridVision
            self.vision = HybridVision(models_dir, pose_name=pose_name, adapter=adapter)

    def predict(self, rgb):
        observation = self.vision.observe(rgb)
        posture, scores, reason = self.predictor.predict(observation)
        metrics = {k: float(v) if math.isfinite(v) else None for k, v in observation.metrics.items()}
        return posture, max(scores.values(), default=0.), scores, reason, metrics, observation.pose_box

    def close(self):
        self.vision.close()
        self.predictor.close()
