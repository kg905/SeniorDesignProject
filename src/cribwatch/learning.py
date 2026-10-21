"""Baby-independent splits, validation-only abstention tuning, honest test reports."""

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from .data import read_csv, write_csv
from .features import FEATURES, LABELS, SCHEMA_VERSION, VISION_PREPROCESSING, feature_vector


def wilson(successes, total):
    """95% binomial interval; frame dependence means this is only descriptive."""
    if not total:
        return [None, None]
    z, p = 1.959963984540054, successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [center - half, center + half]


def group_split(rows, seed=42, *, group_field="baby_id"):
    groups = np.array([r[group_field] for r in rows])
    y = np.array([r["posture"] for r in rows])
    if len(set(groups)) < 15:
        raise ValueError(f"At least 15 distinct {group_field} groups required; more are needed for a strong claim")
    x = np.arange(len(rows))
    for attempt in range(100):
        train_val, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                                random_state=seed + attempt).split(x, y, groups))
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                       random_state=seed + attempt).split(train_val, y[train_val], groups[train_val]))
        train, val = train_val[tr], train_val[va]
        if all(all(len(set(groups[part][y[part] == label])) >= 2 for label in LABELS)
               for part in (train, val, test)):
            return train, val, test
    raise ValueError(f"Could not form disjoint {group_field} splits with >=2 groups per class in every split")


def predict_rows(pipeline, rows, threshold):
    x = np.stack([feature_vector(r) for r in rows])
    probabilities = pipeline.predict_proba(x)
    indices = probabilities.argmax(axis=1)
    result = pipeline.classes_[indices].astype(object)
    valid = np.array([str(r.get("pose_ok", "0")) == "1" for r in rows])
    result[(probabilities.max(axis=1) < threshold) | ~valid] = "unknown"
    return result


def summarize(rows, predictions, *, group_field="baby_id"):
    y = np.array([r["posture"] for r in rows])
    predictions = np.asarray(predictions)
    accepted = predictions != "unknown"
    correct = predictions == y
    unsafe = y != "back"
    n = len(y)
    # Cluster bootstrap samples the declared groups, not correlated augmented images.
    group_ids = sorted({r[group_field] for r in rows})
    group_stats = []
    for g in group_ids:
        m = np.array([r[group_field] == g for r in rows])
        group_stats.append([int(correct[m].sum()), int(m.sum())])
    rng = np.random.default_rng(42)
    stats = np.array(group_stats)
    boot = stats[rng.integers(len(stats), size=(2000, len(stats)))].sum(axis=1)
    report = {
        "n_images": n, "n_babies": len(group_ids) if group_field == "baby_id" else None,
        "grouping_field": group_field, "n_evaluation_groups": len(group_ids),
        "accuracy_all_images_unknown_counts_as_error": float(correct.mean()),
        "accuracy_95ci_group_cluster_bootstrap": np.quantile(boot[:, 0] / boot[:, 1], [.025, .975]).tolist(),
        "coverage": float(accepted.mean()),
        "accuracy_on_accepted_only": float(correct[accepted].mean()) if accepted.any() else None,
        "unsafe_predicted_back_count": int(((predictions == "back") & unsafe).sum()),
        "unsafe_total": int(unsafe.sum()),
        "class_order": [*LABELS, "unknown"],
        "confusion_matrix": confusion_matrix(y, predictions, labels=[*LABELS, "unknown"]).tolist(),
        "per_class": {},
        "limitations": "Research posture labels only, not SIDS/airway outcomes. Intervals are uncertain with few babies. No event latency or false-alarm/hour claim from still images.",
    }
    for label in LABELS:
        m = y == label
        count, total = int(correct[m].sum()), int(m.sum())
        report["per_class"][label] = {"correct": count, "total": total, "recall": count / total if total else None,
                                       "frame_binomial_95ci_descriptive_only": wilson(count, total)}
    report["posture_95pct_point_estimates_met"] = bool(correct.mean() >= .95 and all(
        v["recall"] is not None and v["recall"] >= .95 for v in report["per_class"].values()))
    report["research_95pct_point_target_met"] = report["posture_95pct_point_estimates_met"] and group_field == "baby_id"
    if group_field == "baby_id":
        report["accuracy_95ci_baby_cluster_bootstrap"] = report["accuracy_95ci_group_cluster_bootstrap"]
    else:
        report["limitations"] += " No infant/session IDs: source-image groups cannot demonstrate accuracy on unseen babies."
    return report


def train(features_csv, output_dir, seed=42, *, exploratory=False, min_validation_coverage=0.5):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new run directory to preserve the test report")
    all_rows = read_csv(features_csv)
    if not 0 <= min_validation_coverage <= 1:
        raise ValueError("Minimum validation coverage must be between 0 and 1")
    if exploratory:
        rows = [r for r in all_rows if r.get("training_eligible") == "1" and r.get("posture") in LABELS
                and r.get("label_source") in {"dataset_annotation", "human_review"}]
        if any(not r.get("source_group", "").strip() for r in rows):
            raise ValueError("Exploratory training requires audited source-image groups")
        # Retain one copy of each exact file, without promoting inferred labels.
        seen, unique = set(), []
        for row in rows:
            if row["sha256"] not in seen:
                seen.add(row["sha256"])
                unique.append(row)
        rows = unique
        group_field = "source_group"
    else:
        rows = [r for r in all_rows if r.get("reviewed") == "1" and r.get("posture") in LABELS
                and r.get("label_conflict", "0") != "1"]
        group_field = "baby_id"
    if not rows:
        raise ValueError("No reviewed posture ground truth; dataset detection labels are not posture labels")
    if not exploratory and any(not r.get("baby_id", "").strip() or not r.get("session_id", "").strip() for r in rows):
        raise ValueError("Every labeled row requires real baby_id and session_id, not filenames or dataset names")
    if any(not r.get("sha256", "").strip() for r in rows):
        raise ValueError("Each row needs its image SHA256")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image hashes found; reconcile labels and deduplicate before splitting")
    train_idx, val_idx, test_idx = group_split(rows, seed, group_field=group_field)
    parts = [[rows[i] for i in idx] for idx in (train_idx, val_idx, test_idx)]
    fit_rows = [r for r in parts[0] if r.get("pose_ok") == "1"]
    if set(r["posture"] for r in fit_rows) != set(LABELS):
        raise ValueError("Training pose-quality gate removed a class; improve the pose extractor or collect suitable images")
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                          RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                 class_weight="balanced_subsample", n_jobs=2, random_state=seed))
    model.fit(np.stack([feature_vector(r) for r in fit_rows]), [r["posture"] for r in fit_rows])
    val_y = np.array([r["posture"] for r in parts[1]])
    candidates = []
    for threshold in np.linspace(.5, 1.0, 21):
        pred = predict_rows(model, parts[1], float(threshold))
        unsafe_as_back = int(((val_y != "back") & (pred == "back")).sum())
        wrong = int(((pred != val_y) & (pred != "unknown")).sum())
        accepted = int((pred != "unknown").sum())
        candidates.append((unsafe_as_back, wrong, -accepted, float(threshold)))
    # Do not tune to a vacuous all-abstaining classifier merely to get zero errors.
    qualifying = [c for c in candidates if -c[2] / len(parts[1]) >= min_validation_coverage]
    threshold = min(qualifying)[3] if qualifying else min(candidates, key=lambda c: (c[2], c[0], c[1], c[3]))[3]
    artifact = {"schema_version": SCHEMA_VERSION, "features": list(FEATURES), "labels": list(LABELS),
                "pipeline": model, "threshold": threshold, "research_only": True,
                "exploratory": exploratory, "evaluation_grouping": group_field,
                "preprocessing": VISION_PREPROCESSING,
                "training_sha256": hashlib.sha256(Path(features_csv).read_bytes()).hexdigest()}
    validation = summarize(parts[1], predict_rows(model, parts[1], threshold), group_field=group_field)
    test_predictions = predict_rows(model, parts[2], threshold)
    test_report = summarize(parts[2], test_predictions, group_field=group_field)
    for part, report in [(parts[1], validation), (parts[2], test_report)]:
        report["exploratory"] = exploratory
        report["score_threshold"] = threshold
        report["requested_min_validation_coverage"] = min_validation_coverage
        report["validation_coverage_constraint_met"] = bool(qualifying)
        report["pose_gate_coverage_ceiling"] = sum(r.get("pose_ok") == "1" for r in part) / len(part)
    test_report["unthresholded_diagnostic"] = summarize(parts[2], predict_rows(model, parts[2], 0), group_field=group_field)
    ranges = {}
    for label in LABELS:
        x = np.stack([feature_vector(r) for r in fit_rows if r["posture"] == label])
        ranges[label] = {}
        for i, name in enumerate(FEATURES):
            finite = x[:, i][np.isfinite(x[:, i])]
            ranges[label][name] = np.quantile(finite, [.05, .5, .95]).tolist() if len(finite) else None
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / "posture.joblib")
    for filename, obj in [("validation.json", validation), ("test.json", test_report),
                          ("training_feature_percentiles.json", {"note": "5th/50th/95th descriptive training percentiles; NOT medical safe ranges", "classes": ranges})]:
        (out / filename).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    split_rows = []
    for name, part in zip(("train", "validation", "test"), parts):
        split_rows.extend(dict(r, split=name) for r in part)
    write_csv(out / "split_manifest.csv", split_rows, list(rows[0]) + ["split"])
    write_csv(out / "test_predictions.csv", [dict(r, predicted=p) for r, p in zip(parts[2], test_predictions)],
              list(rows[0]) + ["predicted"])
    return test_report


class Predictor:
    def __init__(self, path, *, allow_exploratory=False, models_dir="models"):
        # Joblib can execute Python: load only a model YOU trained or otherwise trust.
        self.artifact = joblib.load(path)
        if self.artifact.get("schema_version") != SCHEMA_VERSION or self.artifact.get("features") != list(FEATURES):
            raise ValueError("Incompatible feature schema")
        if self.artifact.get("preprocessing") != VISION_PREPROCESSING:
            raise ValueError("Incompatible image preprocessing")
        if self.artifact.get("exploratory") and not allow_exploratory:
            raise ValueError("Exploratory source-image model is for offline research only; use analyze-video --allow-exploratory, not live monitoring")
        self.pipeline = self.artifact["pipeline"]
        self.threshold = self.artifact["threshold"]
        self.models_dir = models_dir
        self.embedding_extractor = None

    def predict(self, observation, *, rgb=None):
        if self.artifact.get("input_kind") == "image_metrics":
            from .image_model import EmbeddingExtractor, combined_vector
            if rgb is None:
                raise ValueError("Image+metric model requires an RGB frame")
            if observation.reason in {"multiple_people", "multiple_faces"}:
                return "unknown", {}, observation.reason
            if np.std(rgb.astype(float), axis=(0, 1)).mean() < 3:
                return "unknown", {}, "blank_or_flat_image"
            if self.embedding_extractor is None:
                self.embedding_extractor = EmbeddingExtractor(self.models_dir)
            row = dict(observation.metrics, pose_ok=int(observation.pose_ok))
            vector = combined_vector(row, self.embedding_extractor.embed(rgb))
        else:
            if not observation.pose_ok:
                return "unknown", {}, observation.reason
            vector = feature_vector(observation.metrics)
        p = self.pipeline.predict_proba([vector])[0]
        scores = dict(zip(self.pipeline.classes_.tolist(), p.tolist()))
        posture = str(self.pipeline.classes_[int(p.argmax())]) if max(p) >= self.threshold else "unknown"
        return posture, scores, "ok" if posture != "unknown" else "low_model_score"

    def close(self):
        if self.embedding_extractor is not None:
            self.embedding_extractor.close()
            self.embedding_extractor = None


def suggest_labels(features_csv, output):
    """Bootstrap suggestions from a reviewed seed, with NO automatic ground truth.

    Suggestions are label-assistance only and do not count as validation. Do not
    use this after the final split is frozen: that could influence test labeling.
    """
    if Path(output).exists():
        raise FileExistsError("Use a new suggestion filename")
    rows = read_csv(features_csv)
    seed = [r for r in rows if r.get("reviewed") == "1" and r.get("posture") in LABELS and r.get("pose_ok") == "1"]
    if any(sum(r["posture"] == label for r in seed) < 3 for label in LABELS):
        raise ValueError("Review at least 3 usable examples of EACH posture before requesting suggestions")
    pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                             RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=2, random_state=42))
    pipeline.fit(np.stack([feature_vector(r) for r in seed]), [r["posture"] for r in seed])
    for row in rows:
        row["suggested_posture"], row["suggestion_score"] = "unknown", ""
        if row.get("reviewed") != "1" and row.get("pose_ok") == "1":
            p = pipeline.predict_proba([feature_vector(row)])[0]
            row["suggested_posture"] = str(pipeline.classes_[int(p.argmax())])
            row["suggestion_score"] = str(float(p.max()))
    # Group proposals so visually similar posture batches can be reviewed together.
    rows.sort(key=lambda r: (r.get("reviewed") == "1", r["suggested_posture"], -float(r["suggestion_score"] or 0)))
    write_csv(output, rows, list(rows[0]))
    return sum(r.get("reviewed") != "1" and r["suggested_posture"] != "unknown" for r in rows)
