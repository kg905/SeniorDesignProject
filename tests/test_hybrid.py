import json
import math
from types import SimpleNamespace as P

import numpy as np
import pytest

from cribwatch.assets import YOLO_ASSETS
from cribwatch.features import FEATURES, Observation, unknown
from cribwatch.hybrid_vision import HybridVision, validate_adapter
from cribwatch.learning import _adapter_metadata
from cribwatch.vision import Vision


def adapter():
    return dict(kind="yolo_mediapipe_roi_v1", yolo_name="yolov8n-pose.pt",
                yolo_sha256=YOLO_ASSETS["yolov8n-pose.pt"][1], confidence=.25, padding=.2)


def test_crop_landmarks_are_remapped_before_original_metric_geometry():
    points = [P(x=.5, y=.2, z=0., visibility=.9, presence=.9) for _ in range(33)]
    for i, x, y in [(11, .2, .3), (12, .8, .3), (23, .3, .7), (24, .7, .7)]:
        points[i] = P(x=x, y=y, z=0., visibility=.9, presence=.9)
    vision = object.__new__(Vision)
    vision.mp = P(Image=lambda **kwargs: kwargs, ImageFormat=P(SRGB=1))
    vision.pose = P(detect=lambda image: P(pose_landmarks=[points], pose_world_landmarks=[]))
    vision.face = P(detect=lambda image: P(face_landmarks=[]))
    result = vision.observe_region(np.zeros((640, 640, 3), np.uint8), (.25, .1, .75, .9))
    assert result.pose_ok and len(result.metrics) == 12
    assert result.metrics["shoulder_vec_mag"] == pytest.approx(.3 / .32)
    assert points[11].x == .2  # Underlying detector output wasn't mutated.
    assert result.pose_box[0] > .25 and result.pose_box[2] < .75


def make_hybrid(baseline, boxes):
    vision = object.__new__(HybridVision)
    vision.adapter = adapter()
    seen = []
    recovered = Observation(dict.fromkeys(FEATURES, .5), True, "ok")
    vision.mediapipe = P(observe=lambda rgb: baseline,
                        observe_region=lambda rgb, region: (seen.append(region), recovered)[1])
    def predict(**kwargs):
        seen.append(kwargs["source"][0, 0].tolist())
        return [P(boxes=P(xyxy=P(cpu=lambda: P(numpy=lambda: np.array(boxes)))))]
    vision.yolo = P(predict=predict)
    return vision, seen, recovered


def test_yolo_recovers_crop_with_rgb_bgr_conversion_and_clamped_padding():
    vision, seen, recovered = make_hybrid(unknown("no_pose"), [[0, 100, 320, 500]])
    rgb = np.full((640, 640, 3), [255, 0, 0], np.uint8)
    assert vision.observe(rgb) is recovered
    assert seen[0] == [0, 0, 255]
    assert np.allclose(seen[1], [0, 20 / 640, 384 / 640, 580 / 640])
    assert vision.last_route == "yolo_crop"


def test_good_full_frame_pose_skips_yolo_and_multiple_bodies_are_unknown():
    good = Observation(dict.fromkeys(FEATURES, .5), True, "ok")
    vision, seen, _ = make_hybrid(good, [[0, 0, 320, 640]])
    assert vision.observe(np.zeros((640, 640, 3), np.uint8)) is good
    assert seen == []
    vision, seen, _ = make_hybrid(unknown("no_pose"), [[0, 0, 300, 600], [320, 0, 640, 600]])
    assert vision.observe(np.zeros((640, 640, 3), np.uint8)).reason == "multiple_people"
    assert len(seen) == 1  # No crop selected from multiple bodies.


def test_no_yolo_body_preserves_unknown_and_adapter_cannot_mix_in_training():
    baseline = unknown("no_pose")
    vision, seen, _ = make_hybrid(baseline, [])
    assert vision.observe(np.zeros((640, 640, 3), np.uint8)) is baseline
    encoded = json.dumps(adapter(), sort_keys=True)
    assert _adapter_metadata([{"vision_adapter": encoded}]) == adapter()
    with pytest.raises(ValueError, match="Mixed vision adapters"):
        _adapter_metadata([{"vision_adapter": encoded}, {"vision_adapter": ""}])
    invalid = adapter() | {"yolo_sha256": "0" * 64}
    with pytest.raises(ValueError, match="Pinned YOLO"):
        validate_adapter(invalid)


def test_keypoint_alternative_uses_dense_face_count_and_never_invents_head_depth():
    vision, _, _ = make_hybrid(unknown("no_pose"), [])
    vision.adapter = adapter() | {"kind": "yolo_mediapipe_keypoints_v1"}
    vision.mediapipe.observe_region = lambda rgb, region: unknown("no_pose")
    vision.mediapipe.face_summary = lambda rgb: (1, 478)
    xy = np.full((1, 17, 2), [320, 140], dtype=float)
    for i, x, y in ((5, 192, 192), (6, 448, 192), (11, 224, 448), (12, 416, 448)):
        xy[0, i] = [x, y]
    def tensor(value):
        return P(cpu=lambda: P(numpy=lambda: value))
    result = P(boxes=P(xyxy=tensor(np.array([[150, 70, 500, 550]]))),
               keypoints=P(xy=tensor(xy), conf=tensor(np.full((1, 17), .9))))
    vision.yolo.predict = lambda **kwargs: [result]
    result = vision.observe(np.zeros((640, 640, 3), np.uint8))
    assert result.pose_ok and len(result.metrics) == 12
    assert result.metrics["shoulder_vec_mag"] == pytest.approx(1.)
    assert result.metrics["hip_vec_mag"] == pytest.approx(.75)
    assert result.metrics["left_shoulder_visibility"] == .9
    assert result.metrics["face_detected"] == 1.
    assert result.metrics["face_landmark_count"] == 478
    assert math.isnan(result.metrics["head_lift_ratio"])
    assert vision.last_route == "yolo_keypoints"


def test_hybrid_export_preserves_one_classifier_and_exact_adapter(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from pathlib import Path
    import shutil
    import zipfile
    from cribwatch.data import write_csv
    from cribwatch.features import LABELS
    from cribwatch.learning import train
    from cribwatch import hybrid_vision, metrics_model
    root = Path(__file__).resolve().parents[1]
    bundle_root = tmp_path / "source"
    models = bundle_root / "models"
    models.mkdir(parents=True)
    for folder in ("src", "deploy", "docs", "training"):
        (bundle_root / folder).symlink_to(root / folder, target_is_directory=True)
    for name in ("pose_landmarker_lite.task", "face_landmarker.task"):
        shutil.copy2(root / "models" / name, models / name)
    weights = models / "synthetic-pose.pt"
    weights.write_bytes(b"synthetic packaging fixture; not a model")
    config = adapter() | {"kind": "yolo_mediapipe_keypoints_v1", "yolo_name": weights.name,
                          "yolo_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    rows = []
    for group in range(20):
        for i, label in enumerate(LABELS):
            row = dict(image_path="synthetic-not-an-image", sha256=f"{group}-{label}", baby_id=f"b-{group}",
                       session_id=f"s-{group}", posture=label, reviewed="1", pose_ok="1",
                       vision_adapter=json.dumps(config, sort_keys=True))
            row.update({name: float(i * 10 + group * .001) for name in FEATURES})
            rows.append(row)
    features = tmp_path / "features.csv"
    write_csv(features, rows, list(rows[0]))
    model_dir = tmp_path / "model"
    train(features, model_dir)
    seen = []
    class FakeHybrid:
        def __init__(self, directory, *, pose_name, adapter):
            seen.append(adapter)
        def close(self):
            pass
    monkeypatch.setattr(hybrid_vision, "HybridVision", FakeHybrid)
    predictor = metrics_model.MetricsPredictor(model_dir / "posture.joblib", models_dir=models, showcase=True)
    assert seen == [config]
    predictor.close()
    spec = importlib.util.spec_from_file_location("build_hybrid_handoff", root / "deploy/single-model/build_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "hybrid.zip"
    module.build(model_dir, archive, root=bundle_root)
    with zipfile.ZipFile(archive) as z:
        assert sum(name.endswith(".joblib") for name in z.namelist()) == 1
        assert f"models/{weights.name}" in z.namelist()
        assert "requirements-hybrid.txt" in z.namelist()
        assert "HYBRID.md" in z.namelist()
        assert "src/cribwatch/hybrid_vision.py" in z.namelist()
