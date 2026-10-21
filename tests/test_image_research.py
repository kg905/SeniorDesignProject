import hashlib
import json

import numpy as np
import pytest

from cribwatch.assets import ASSETS
from cribwatch.data import read_csv, write_csv
from cribwatch.features import unknown
from cribwatch.image_model import EMBEDDING_MODEL, propose_unlabeled, train_image_baseline
from cribwatch.learning import Predictor


def test_image_research_training_proposals_and_offline_only_gate(tmp_path):
    from test_learning import synthetic_rows
    rows = synthetic_rows()
    rng = np.random.default_rng(7)
    embeddings = []
    for i, row in enumerate(rows):
        label = {"back": 0, "side": 1, "prone": 2}[row["posture"]]
        row.update(source_group=row["baby_id"], baby_id="", session_id="", reviewed="0",
                   label_source="dataset_annotation", training_eligible="1", quality_reason="ok")
        vector = rng.normal(0, .01, 1024).astype(np.float32)
        vector[label] = 1
        embeddings.append(vector)
    for row in rows[-9:]:
        row.update(posture="unknown", training_eligible="0", reviewed="0", human_verified="0")
    source = tmp_path / "features.csv"
    write_csv(source, rows, list(rows[0]) + ["human_verified"])
    cache = tmp_path / "embeddings.npz"
    np.savez_compressed(cache, sha256=np.array([r["sha256"] for r in rows]), embeddings=np.stack(embeddings),
                        asset_sha256=np.array(ASSETS[EMBEDDING_MODEL][1]))
    run = tmp_path / "run"
    report = train_image_baseline(source, cache, run)
    assert report["n_babies"] is None and not report["research_95pct_point_target_met"]
    with pytest.raises(ValueError, match="offline research"):
        Predictor(run / "posture.joblib")
    predictor = Predictor(run / "posture.joblib", allow_exploratory=True, models_dir="not-needed-for-blank")
    assert predictor.predict(unknown("no_pose"), rgb=np.zeros((32, 32, 3), dtype=np.uint8))[0] == "unknown"
    output = tmp_path / "proposals.csv"
    result = propose_unlabeled(source, cache, run / "posture.joblib", output)
    assert result["images"] == 9
    assert all(r["posture"] == "unknown" and r["reviewed"] == "0" for r in read_csv(output))
    assert all(r["human_verified"] == "0" for r in read_csv(output))
