"""Preplanned baseline versus YOLO ROI recovery; preserve both trained exports."""

import hashlib
import json
from pathlib import Path
import shutil

from .assets import POSE_VARIANTS, YOLO_ASSETS, download_models, download_pose_variant
from .data import extract_manifest, read_csv
from .hybrid_vision import validate_adapter
from .learning import train, validation_key
from .optimize import _check_manifest, diagnose_features


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _train_or_load(features, directory):
    path = directory / "test.json"
    if path.exists():
        report = json.loads(path.read_text())
        if report.get("features_sha256") != _sha(features) or report.get("model_sha256") != _sha(directory / "posture.joblib"):
            raise ValueError("Saved alternative model/report do not match the features")
        return report
    return train(features, directory, exploratory=True, search=True)


def run_hybrid_comparison(manifest, output, models_dir, *, existing_features=None,
                          pose_variant="lite", yolo_weights=None, keypoint_fallback=True):
    manifest, output, models_dir = Path(manifest), Path(output), Path(models_dir)
    pose_name = POSE_VARIANTS[pose_variant]
    rows = read_csv(manifest)
    if not rows:
        raise ValueError("Empty manifest")
    output.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    if yolo_weights is None:
        yolo_name = "yolov8n-pose.pt"
        download_models(models_dir, names=[yolo_name])
        yolo_digest = YOLO_ASSETS[yolo_name][1]
    else:
        source = Path(yolo_weights).resolve()
        yolo_name, yolo_digest = source.name, _sha(source)
        target = models_dir / yolo_name
        if target.exists() and _sha(target) != yolo_digest:
            raise ValueError("Different YOLO weights have the same filename; use a new basename")
        if source != target.resolve():
            shutil.copy2(source, target)
    adapter = validate_adapter({"kind": "yolo_mediapipe_keypoints_v1" if keypoint_fallback else "yolo_mediapipe_roi_v1", "yolo_name": yolo_name,
        "yolo_sha256": yolo_digest, "confidence": .25, "padding": .2})
    config = {"manifest_sha256": _sha(manifest), "pose_variant": pose_variant, "adapter": adapter,
        "existing_features_sha256": _sha(existing_features) if existing_features else None}
    config_path = output / "hybrid_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Hybrid experiment inputs changed; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2))
    baseline_features = output / "features-baseline.csv"
    if existing_features and not baseline_features.exists():
        existing = read_csv(existing_features)
        _check_manifest(rows, existing)
        if (all(r.get("pose_model_name", POSE_VARIANTS["lite"]) == pose_name for r in existing)
                and all(not r.get("vision_adapter") for r in existing)):
            shutil.copy2(existing_features, baseline_features)
            print("Reusing completed baseline features.", flush=True)
        else:
            print("Existing features use another extractor; generating a matching baseline separately.", flush=True)
    if not baseline_features.exists():
        download_pose_variant(models_dir, pose_variant)
        extract_manifest(manifest, baseline_features, models_dir, pose_name=pose_name)
    else:
        # Hybrid inference still needs the same verified assets even when baseline
        # CSVs were produced in an earlier run.
        download_pose_variant(models_dir, pose_variant)
    hybrid_features = output / "features-hybrid.csv"
    if not hybrid_features.exists():
        print("\nExtracting YOLO + MediaPipe alternative...", flush=True)
        extract_manifest(manifest, hybrid_features, models_dir, pose_name=pose_name, adapter=adapter)
    results = {}
    for name, features in (("baseline", baseline_features), ("hybrid", hybrid_features)):
        features_rows = read_csv(features)
        _check_manifest(rows, features_rows)
        print(f"\n=== Train {name}: same labels, source groups, split seed and 12 metrics ===", flush=True)
        model_dir = output / f"{name}-model"
        report = _train_or_load(features, model_dir)
        validation = json.loads((model_dir / "validation.json").read_text())
        diagnostic = diagnose_features(features_rows)
        eligible = [r for r in features_rows if r.get("training_eligible") == "1"]
        for route in ("yolo_crop", "yolo_keypoints"):
            diagnostic[f"usable_{route}_recoveries"] = sum(r.get("pose_ok") == "1" and r.get("extraction_route") == route for r in eligible)
        results[name] = {"model_dir": str(model_dir), "features": str(features),
            "diagnostics": diagnostic, "validation": validation, "test": report}
    recommended = max(results, key=lambda name: validation_key(results[name]["validation"]))
    comparison = {"variants": results, "recommended_by_validation_only": recommended,
        "limitations": "Two preplanned alternatives, same exploratory pilot split. Test results are descriptive; no final unseen-infant, Pi-speed, or 95% claim.",
        "adapter": adapter, "pose_variant": pose_variant}
    comparison_path = output / "hybrid_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    for name, info in results.items():
        print(name, "pose coverage:", f"{info['diagnostics']['pose_coverage']:.1%}",
            "test accuracy:", f"{info['test']['accuracy_all_images_unknown_counts_as_error']:.1%}", flush=True)
    print("Recommended using VALIDATION only:", recommended, flush=True)
    print("Both models preserved; the hybrid is an alternative even if it does not win.", flush=True)
    return {"model_dir": results["hybrid"]["model_dir"], "features": str(hybrid_features),
        "report": results["hybrid"]["test"], "baseline_model_dir": results["baseline"]["model_dir"],
        "comparison_path": str(comparison_path), "comparison": comparison}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", default="models")
    parser.add_argument("--features", help="Existing baseline CSV to reuse when it matches")
    parser.add_argument("--pose-variant", choices=tuple(POSE_VARIANTS), default="lite")
    parser.add_argument("--yolo-weights", help="Optional trusted compatible infant pose checkpoint")
    parser.add_argument("--crop-only", action="store_true")
    args = parser.parse_args()
    result = run_hybrid_comparison(args.manifest, args.output, args.models, existing_features=args.features,
        pose_variant=args.pose_variant, yolo_weights=args.yolo_weights, keypoint_fallback=not args.crop_only)
    print("Hybrid model:", result["model_dir"])
    print("Comparison:", result["comparison_path"])
