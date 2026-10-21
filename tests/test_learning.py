import json

import numpy as np
import pytest

from cribwatch.data import write_csv
from cribwatch.features import FEATURES, LABELS, Observation
from cribwatch.learning import Predictor, group_split, summarize, suggest_labels, train


def synthetic_rows():
    # Deliberately artificial data verifies plumbing, never infant/model accuracy.
    rng = np.random.default_rng(7)
    rows = []
    for baby in range(20):
        for label_i, label in enumerate(LABELS):
            for frame in range(6):
                row = {"image_path": "synthetic-not-an-image", "sha256": f"{baby}-{label}-{frame}",
                       "baby_id": f"baby-{baby}", "session_id": f"session-{baby}",
                       "posture": label, "face_state": "unknown", "reviewed": "1", "pose_ok": "1"}
                row.update(zip(FEATURES, rng.normal(label_i * 10, .05, len(FEATURES)).tolist()))
                rows.append(row)
    return rows


def test_group_split_has_no_baby_leakage():
    rows = synthetic_rows()
    groups = [{rows[i]["baby_id"] for i in split} for split in group_split(rows)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_unknown_is_error_in_overall_accuracy_and_not_hidden():
    rows = synthetic_rows()[:3]
    report = summarize(rows, ["unknown", "back", "unknown"])
    assert report["accuracy_all_images_unknown_counts_as_error"] == pytest.approx(1 / 3)
    assert report["coverage"] == pytest.approx(1 / 3)
    assert report["accuracy_on_accepted_only"] == 1
    assert not report["research_95pct_point_target_met"]


def test_full_training_serialization_and_unknown_gate(tmp_path):
    rows = synthetic_rows()
    file = tmp_path / "features.csv"
    write_csv(file, rows, list(rows[0]))
    run = tmp_path / "run"
    result = train(file, run)
    assert result["n_images"] > 0
    assert set(json.loads((run / "test.json").read_text())["per_class"]) == set(LABELS)
    predictor = Predictor(run / "posture.joblib")
    metrics = {k: float(rows[0][k]) for k in FEATURES}
    assert predictor.predict(Observation(metrics, True, "ok"))[0] == "back"
    assert predictor.predict(Observation(metrics, False, "no_pose"))[0] == "unknown"
    with pytest.raises(FileExistsError):
        train(file, run)


def test_training_refuses_unreviewed_and_duplicate_data(tmp_path):
    rows = synthetic_rows()
    for r in rows:
        r["reviewed"] = "0"
    file = tmp_path / "features.csv"
    write_csv(file, rows, list(rows[0]))
    with pytest.raises(ValueError, match="ground truth"):
        train(file, tmp_path / "run")
    for r in rows:
        r["reviewed"] = "1"
    rows[1]["sha256"] = rows[0]["sha256"]
    write_csv(file, rows, list(rows[0]))
    with pytest.raises(ValueError, match="Duplicate"):
        train(file, tmp_path / "run")


def test_suggestions_never_create_ground_truth(tmp_path):
    from cribwatch.data import read_csv
    rows = synthetic_rows()
    for row in rows[-18:]:
        row["reviewed"], row["posture"] = "0", "unknown"
    source, output = tmp_path / "features.csv", tmp_path / "suggestions.csv"
    write_csv(source, rows, list(rows[0]))
    assert suggest_labels(source, output) == 18
    pending = [r for r in read_csv(output) if r["reviewed"] != "1"]
    assert len(pending) == 18
    assert all(r["posture"] == "unknown" for r in pending)
    assert all(r["suggested_posture"] in LABELS for r in pending)


def test_exploratory_groups_never_claim_infant_accuracy_or_enable_live_model(tmp_path):
    rows = synthetic_rows()
    for row in rows:
        row.update(source_group=row.pop("baby_id"), baby_id="", session_id="", reviewed="0",
                   label_source="dataset_annotation", training_eligible="1")
    source = tmp_path / "features.csv"
    write_csv(source, rows, list(rows[0]))
    run = tmp_path / "exploratory"
    result = train(source, run, exploratory=True)
    assert result["n_babies"] is None
    assert result["grouping_field"] == "source_group"
    assert not result["research_95pct_point_target_met"]
    with pytest.raises(ValueError, match="offline research"):
        Predictor(run / "posture.joblib")
    predictor = Predictor(run / "posture.joblib", allow_exploratory=True)
    assert predictor.artifact["exploratory"]
    with pytest.raises(ValueError, match="ground truth"):
        train(source, tmp_path / "independent")
