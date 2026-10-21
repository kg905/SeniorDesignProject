"""Exploratory image-embedding + metric baseline; offline research only.

Generic ImageNet embeddings can help when infant pose landmarks are absent, but
do not verify that a baby is present, asleep, breathing, or has a clear airway.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .assets import ASSETS
from .data import read_csv, write_csv
from .features import FEATURES, LABELS, SCHEMA_VERSION, VISION_PREPROCESSING, feature_vector

EMBEDDING_MODEL = "mobilenet_v3_small.tflite"


class EmbeddingExtractor:
    def __init__(self, models):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model = Path(models) / EMBEDDING_MODEL
        if hashlib.sha256(model.read_bytes()).hexdigest() != ASSETS[EMBEDDING_MODEL][1]:
            raise ValueError("Image embedding model checksum mismatch")
        self.mp = mp
        self.task = vision.ImageEmbedder.create_from_options(vision.ImageEmbedderOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)), l2_normalize=True, quantize=False))

    def embed(self, rgb):
        import cv2
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB image")
        rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        result = self.task.embed(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return np.asarray(result.embeddings[0].embedding, dtype=np.float32)

    def close(self):
        self.task.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def extract_embeddings(manifest, models, output):
    if Path(output).exists():
        raise FileExistsError("Use a new embedding output path")
    rows, by_hash, vectors = read_csv(manifest), {}, []
    with EmbeddingExtractor(models) as extractor:
        for i, row in enumerate(rows):
            digest = row["sha256"]
            path = Path(row["image_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Image changed since audit")
            if digest not in by_hash:
                with Image.open(path) as im:
                    rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
                by_hash[digest] = extractor.embed(rgb)
            vectors.append(by_hash[digest])
            if (i + 1) % 500 == 0:
                print(f"Embedded {i + 1}/{len(rows)} images", flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, sha256=np.array([r["sha256"] for r in rows]), embeddings=np.stack(vectors),
                        asset_sha256=np.array(ASSETS[EMBEDDING_MODEL][1]),
                        input_sha256=np.array(hashlib.sha256(Path(manifest).read_bytes()).hexdigest()))
    return len(rows)


def combined_vector(row, embedding):
    return np.concatenate([embedding, feature_vector(row), [float(str(row.get("pose_ok", "0")) == "1")]])


def train_image_baseline(features_csv, embeddings_npz, output_dir, seed=42):
    from .learning import group_split, summarize
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new run directory")
    all_rows = read_csv(features_csv)
    with np.load(embeddings_npz, allow_pickle=False) as cache:
        if str(cache["asset_sha256"]) != ASSETS[EMBEDDING_MODEL][1]:
            raise ValueError("Embedding asset checksum does not match")
        embeddings = dict(zip(cache["sha256"].tolist(), cache["embeddings"]))
        if any(row["sha256"] not in embeddings for row in all_rows):
            raise ValueError("Embedding cache does not cover all input image hashes")
    seen = set()
    rows = []
    for row in all_rows:
        if row.get("training_eligible") == "1" and row.get("posture") in LABELS and row["sha256"] not in seen:
            rows.append(row)
            seen.add(row["sha256"])
    train_i, val_i, test_i = group_split(rows, seed, group_field="source_group")
    x = np.stack([combined_vector(row, embeddings[row["sha256"]]) for row in rows])
    y = np.array([r["posture"] for r in rows])
    counts = Counter(rows[i]["source_group"] for i in train_i)
    weights = np.array([1 / counts[rows[i]["source_group"]] for i in train_i])
    candidates = []
    for c in [.1, 1., 10.]:
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                              StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000))
        model.fit(x[train_i], y[train_i], logisticregression__sample_weight=weights)
        score = balanced_accuracy_score(y[val_i], model.predict(x[val_i]))
        candidates.append((float(score), f"logistic_C_{c}", model))
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                               max_features="sqrt", n_jobs=2, random_state=seed))
    model.fit(x[train_i], y[train_i], extratreesclassifier__sample_weight=weights)
    score = balanced_accuracy_score(y[val_i], model.predict(x[val_i]))
    candidates.append((float(score), "extra_trees", model))
    _, selected, model = max(candidates, key=lambda c: c[0])
    # Tune rejection only on validation. Preserve at least 50% coverage when possible.
    p = model.predict_proba(x[val_i]); labels = model.classes_[p.argmax(1)]
    choices = []
    for threshold in np.linspace(.34, .9, 29):
        accepted = p.max(1) >= threshold
        if accepted.mean() >= .5:
            errors_as_back = ((labels == "back") & (y[val_i] != "back") & accepted).sum()
            errors = ((labels != y[val_i]) & accepted).sum()
            choices.append((int(errors_as_back), int(errors), -int(accepted.sum()), float(threshold)))
    threshold = min(choices)[3] if choices else .34

    def report(indices, threshold):
        probs = model.predict_proba(x[indices])
        pred = model.classes_[probs.argmax(1)].astype(object)
        pred[probs.max(1) < threshold] = "unknown"
        rejected = np.array([rows[i].get("quality_reason") in {"multiple_people", "multiple_faces"} for i in indices])
        pred[rejected] = "unknown"
        result = summarize([rows[i] for i in indices], pred, group_field="source_group")
        result.update(exploratory=True, input_kind="image_metrics", selected_head=selected, score_threshold=threshold,
                      limitations=result["limitations"] + " Generic frozen ImageNet features; no baby-presence/airway detector. Offline research only.")
        return result, pred
    validation, _ = report(val_i, threshold)
    test, predictions = report(test_i, threshold)
    test["unthresholded_diagnostic"] = report(test_i, 0)[0]
    artifact = {"schema_version": SCHEMA_VERSION, "features": list(FEATURES), "labels": list(LABELS),
                "pipeline": model, "threshold": threshold, "research_only": True, "exploratory": True,
                "input_kind": "image_metrics", "preprocessing": VISION_PREPROCESSING,
                "embedding_model": EMBEDDING_MODEL, "embedding_sha256": ASSETS[EMBEDDING_MODEL][1],
                "training_sha256": hashlib.sha256(Path(features_csv).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "posture.joblib")
    for name, result in [("validation.json", validation), ("test.json", test),
                         ("head_selection.json", [{"balanced_validation_accuracy": c[0], "head": c[1]} for c in candidates])]:
        (out / name).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    split_rows = [dict(rows[i], split=name) for name, idx in zip(["train", "validation", "test"], [train_i, val_i, test_i]) for i in idx]
    write_csv(out / "split_manifest.csv", split_rows, list(rows[0]) + ["split"])
    write_csv(out / "test_predictions.csv", [dict(rows[i], predicted=p) for i, p in zip(test_i, predictions)], list(rows[0]) + ["predicted"])
    return test


def propose_unlabeled(features_csv, embeddings_npz, classifier, output):
    """Research suggestions for review, not new posture ground truth."""
    if Path(output).exists():
        raise FileExistsError("Use a new review queue output")
    artifact = joblib.load(classifier)  # Only your own trusted trained artifact.
    if artifact.get("input_kind") != "image_metrics" or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected current image+metric artifact")
    with np.load(embeddings_npz, allow_pickle=False) as cache:
        if str(cache["asset_sha256"]) != artifact.get("embedding_sha256"):
            raise ValueError("Embedding model mismatch")
        vectors = dict(zip(cache["sha256"].tolist(), cache["embeddings"]))
    rows = [r for r in read_csv(features_csv) if r.get("posture") == "unknown"]
    x = np.stack([combined_vector(r, vectors[r["sha256"]]) for r in rows])
    model = artifact["pipeline"]
    probabilities = model.predict_proba(x)
    for row, p in zip(rows, probabilities):
        accepted = p.max() >= artifact["threshold"] and row.get("quality_reason") not in {"multiple_people", "multiple_faces"}
        row.update(suggested_posture=str(model.classes_[p.argmax()]) if accepted else "unknown",
                   suggestion_score=float(p.max()), suggestion_source="exploratory_image_model",
                   reviewed="0", human_verified="0")
        # row['posture'] deliberately remains unknown.
    rows.sort(key=lambda r: (r["suggested_posture"], -r["suggestion_score"]))
    write_csv(output, rows, list(rows[0]) if rows else ["image_path", "posture", "suggested_posture"])
    return {"images": len(rows), "proposals": dict(Counter(r["suggested_posture"] for r in rows))}
