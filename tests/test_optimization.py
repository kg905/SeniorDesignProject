import hashlib
import io
import json
import zipfile

import pytest

from cribwatch import assets
from cribwatch.data import read_csv, write_csv
from cribwatch.features import FEATURES, LABELS
from cribwatch.learning import train
from cribwatch.optimize import diagnose_features, run_optimization


def rows_for_search():
    rows = []
    for group in range(20):
        for i, label in enumerate(LABELS):
            row = dict(image_path=f"synthetic-{group}-{label}", sha256=f"synthetic-{group}-{label}",
                baby_id="", session_id="", source_group=f"g-{group}", reviewed="1",
                posture=label, label_source="human_review", training_eligible="1",
                pose_ok="1", quality_reason="ok")
            row.update({name: float(10 * i + .001 * group) for name in FEATURES})
            rows.append(row)
    return rows


def test_pose_variant_records_https_source_and_detects_changed_cache(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("pose_detector.tflite", b"synthetic fixture, not a real model")
    monkeypatch.setattr(assets, "download_models", lambda *args, **kwargs: None)
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(stream.getvalue()))
    name = assets.download_pose_variant(tmp_path, "full")
    receipt = json.loads((tmp_path / (name + ".source.json")).read_text())
    assert receipt["url"].startswith("https://storage.googleapis.com/mediapipe-models/")
    assert receipt["sha256"] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    assert assets.download_pose_variant(tmp_path, "full") == name
    (tmp_path / name).write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="Unverified existing"):
        assets.download_pose_variant(tmp_path, "full")


def test_asset_metadata_requires_one_pose_and_preserves_pinned_face_hash():
    with pytest.raises(ValueError, match="one recognized pose"):
        assets.classifier_assets({"vision_assets": {"../other.task": "a" * 64}})
    with pytest.raises(ValueError, match="Pinned"):
        assets.classifier_assets({"vision_assets": {"pose_landmarker_full.task": "a" * 64,
                                                   "face_landmarker.task": "b" * 64}})


def test_validation_search_does_not_use_test_metrics(tmp_path):
    rows = rows_for_search()
    features = tmp_path / "features.csv"
    write_csv(features, rows, list(rows[0]))
    first = tmp_path / "first"
    a = train(features, first, exploratory=True, search=True, validation_only=True)
    assert not (first / "test.json").exists()
    assert not (first / "test_predictions.csv").exists()
    assert len(json.loads((first / "classifier_search.json").read_text())["candidates"]) == 7
    held_out = {r["sha256"] for r in read_csv(first / "split_manifest.csv") if r["split"] == "test"}
    for row in rows:
        if row["sha256"] in held_out:
            row.update({name: -9999. for name in FEATURES})
    write_csv(features, rows, list(rows[0]))
    b = train(features, tmp_path / "second", exploratory=True, search=True, validation_only=True)
    assert (a["selected_classifier"], a["score_threshold"], a["per_class"]) == (
        b["selected_classifier"], b["score_threshold"], b["per_class"])


def test_lite_optimization_reuses_features_resumes_and_keeps_one_final_classifier(tmp_path):
    rows = rows_for_search()
    manifest = tmp_path / "manifest.csv"
    original = [{k: v for k, v in r.items() if k not in FEATURES and k not in {"pose_ok", "quality_reason"}} for r in rows]
    write_csv(manifest, original, list(original[0]))
    features = tmp_path / "features.csv"
    write_csv(features, rows, list(rows[0]))
    output = tmp_path / "optimization"
    result = run_optimization(manifest, output, tmp_path / "not-needed", existing_features=features, variants=("lite",))
    final = tmp_path / "optimization/best-model"
    assert len(list(final.glob("*.joblib"))) == 1
    assert (final / "test.json").exists()
    assert not (output / "validation-lite/test.json").exists()
    assert result["report"]["rejection_reasons"]["accepted"] > 0
    resumed = run_optimization(manifest, output, tmp_path / "not-needed", existing_features=features, variants=("lite",))
    assert resumed["report"] == result["report"]
    rows[0]["posture"] = "prone"
    write_csv(features, rows, list(rows[0]))
    with pytest.raises(ValueError, match="inputs changed"):
        run_optimization(manifest, output, tmp_path / "not-needed", existing_features=features, variants=("lite",))


def test_pose_diagnostic_keeps_failed_images_in_denominator():
    rows = rows_for_search()
    rows[0].update(pose_ok="0", quality_reason="no_pose")
    report = diagnose_features(rows)
    assert report["pose_coverage"] == pytest.approx(59 / 60)
    assert report["classes"]["back"]["failure_reasons"] == {"no_pose": 1}


def test_full_extractor_is_bound_to_runtime_and_single_classifier_handoff(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import shutil
    from cribwatch import metrics_model
    root = Path(__file__).resolve().parents[1]
    bundle_root = tmp_path / "source"
    models = bundle_root / "models"
    models.mkdir(parents=True)
    full = models / "pose_landmarker_full.task"
    full.write_bytes(b"synthetic full asset for packaging, not inference")
    shutil.copy2(root / "models/face_landmarker.task", models / "face_landmarker.task")
    for folder in ("src", "deploy", "docs"):
        (bundle_root / folder).symlink_to(root / folder, target_is_directory=True)
    rows = rows_for_search()
    for row in rows:
        row.update(pose_model_name=full.name, pose_model_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
                   face_model_sha256=assets.ASSETS["face_landmarker.task"][1])
    features = tmp_path / "features.csv"
    write_csv(features, rows, list(rows[0]))
    model = tmp_path / "model"
    train(features, model, exploratory=True)
    seen = []
    class FakeVision:
        def __init__(self, directory, *, pose_name):
            seen.append(pose_name)
        def close(self):
            pass
    monkeypatch.setattr(metrics_model, "Vision", FakeVision)
    predictor = metrics_model.MetricsPredictor(model / "posture.joblib", models_dir=models, showcase=True)
    assert seen == [full.name]
    predictor.close()
    spec = importlib.util.spec_from_file_location("build_full_handoff", root / "deploy/single-model/build_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "handoff.zip"
    module.build(model, archive, root=bundle_root)
    with zipfile.ZipFile(archive) as z:
        assert "models/pose_landmarker_full.task" in z.namelist()
        assert "models/pose_landmarker_lite.task" not in z.namelist()
        assert sum(name.endswith(".joblib") for name in z.namelist()) == 1
    full.write_bytes(b"changed asset")
    with pytest.raises(ValueError, match="checksum mismatch"):
        metrics_model.MetricsPredictor(model / "posture.joblib", models_dir=models, showcase=True)
