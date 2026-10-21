"""Offline Colab/video-file analysis. Never sends a physical alarm."""

import json
import math
from pathlib import Path

from .learning import Predictor
from .policy import AlarmPolicy, PolicyConfig
from .vision import Vision


def analyze_video(source, classifier, models, output, *, sample_fps=5, allow_exploratory=False):
    import cv2
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    if Path(output).exists():
        raise FileExistsError("Preserving existing video results; choose another output")
    predictor = Predictor(classifier, allow_exploratory=allow_exploratory, models_dir=models)
    policy = AlarmPolicy(PolicyConfig(max_gap_seconds=max(1.5, 2 / sample_fps)))
    cap = cv2.VideoCapture(str(source))
    count = 0
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video must supply valid frame rate for temporal analysis")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        frame_index, next_time = 0, 0.0
        with Vision(models) as vision, open(output, "x") as log:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                now = frame_index / fps
                frame_index += 1
                if now + 1e-9 < next_time:
                    continue
                next_time = now + 1 / sample_fps
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                observation = vision.observe(rgb)
                posture, scores, reason = predictor.predict(observation, rgb=rgb)
                result = {"video_seconds": now, **policy.update(posture, now), "reason": reason,
                          "model_scores": scores, "metrics": {k: v if math.isfinite(v) else None for k, v in observation.metrics.items()}}
                log.write(json.dumps(result, allow_nan=False) + "\n")
                count += 1
        if not count:
            raise ValueError("Video contained no readable frames")
        return count
    finally:
        cap.release()
        predictor.close()
