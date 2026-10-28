"""Small validation-only search; export exactly one classifier and its extractor."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .assets import POSE_VARIANTS, download_pose_variant
from .data import extract_manifest, read_csv, write_csv
from .features import LABELS
from .learning import train, validation_key


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_manifest(manifest_rows, features):
    if len(manifest_rows) != len(features):
        raise ValueError("Saved features do not match the current manifest")
    for original, extracted in zip(manifest_rows, features):
        if any(str(extracted.get(k, "")) != str(v) for k, v in original.items()):
            raise ValueError("Saved features have different labels, paths, or grouping; preserve the old run")


def diagnose_features(rows):
    eligible = [r for r in rows if r.get("training_eligible") == "1" and r.get("posture") in LABELS]
    result = {"eligible_images": len(eligible), "classes": {}}
    for label in LABELS:
        part = [r for r in eligible if r["posture"] == label]
        usable = sum(r.get("pose_ok") == "1" for r in part)
        result["classes"][label] = {"images": len(part), "usable": usable,
            "pose_coverage": usable / len(part) if part else 0.,
            "failure_reasons": dict(Counter(r.get("quality_reason", "unknown") for r in part if r.get("pose_ok") != "1"))}
    result["pose_coverage"] = sum(r.get("pose_ok") == "1" for r in eligible) / len(eligible) if eligible else 0.
    times = [float(r["inference_ms"]) for r in eligible if r.get("inference_ms") and np.isfinite(float(r["inference_ms"]))]
    result["inference_median_ms_this_machine"] = float(np.median(times)) if times else None
    result["inference_p95_ms_this_machine"] = float(np.quantile(times, .95)) if times else None
    return result


def run_optimization(manifest, output, models_dir, *, existing_features=None,
                     variants=("lite", "full"), seed=42):
    """Keep splits/labels fixed; select only on validation; test the winner once.

    Existing public/pilot test data already viewed remains exploratory. This
    search cannot convert it into a fresh external evaluation.
    """
    manifest, output, models_dir = Path(manifest), Path(output), Path(models_dir)
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or any(v not in POSE_VARIANTS for v in variants):
        raise ValueError("Choose distinct lite/full/heavy variants")
    rows = read_csv(manifest)
    if not rows:
        raise ValueError("Empty manifest")
    output.mkdir(parents=True, exist_ok=True)
    config = {"manifest_sha256": _sha(manifest), "variants": list(variants), "seed": seed,
              "existing_features_sha256": _sha(existing_features) if existing_features else None}
    config_path = output / "search_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Optimization inputs changed; choose a new output directory")
    config_path.write_text(json.dumps(config, indent=2))
    completed, errors = [], {}
    for variant in variants:
        print(f"\n=== Pose extractor: {variant} ===", flush=True)
        features = output / f"features-{variant}.csv"
        try:
            if variant == "lite" and existing_features and not features.exists():
                existing = read_csv(existing_features)
                _check_manifest(rows, existing)
                if any(r.get("pose_model_name", POSE_VARIANTS["lite"]) != POSE_VARIANTS["lite"] for r in existing):
                    raise ValueError("Existing features were not extracted with Lite")
                shutil.copy2(existing_features, features)
                print("Reusing completed Lite features; no image reprocessing.", flush=True)
            if not features.exists():
                pose_name = download_pose_variant(models_dir, variant)
                extract_manifest(manifest, features, models_dir, pose_name=pose_name)
            features_rows = read_csv(features)
            _check_manifest(rows, features_rows)
            diagnostic = diagnose_features(features_rows)
            (output / f"diagnostics-{variant}.json").write_text(json.dumps(diagnostic, indent=2))
            for label, info in diagnostic["classes"].items():
                print(f"{label}: {info['usable']}/{info['images']} usable ({info['pose_coverage']:.1%})", flush=True)
            candidate_dir = output / f"validation-{variant}"
            validation_path = candidate_dir / "validation.json"
            if validation_path.exists():
                validation = json.loads(validation_path.read_text())
                if validation.get("features_sha256") != _sha(features):
                    raise ValueError("Features changed after validation")
            else:
                validation = train(features, candidate_dir, seed, exploratory=True,
                                   search=True, validation_only=True)
            # Only train/validation failures are offered for image review.
            split_rows = read_csv(candidate_dir / "split_manifest.csv")
            failures = [r for r in split_rows if r["split"] != "test" and r.get("pose_ok") != "1"]
            write_csv(output / f"review-failures-{variant}.csv", failures, list(split_rows[0]))
            completed.append({"variant": variant, "features": str(features),
                              "diagnostics": diagnostic, "validation": validation})
        except Exception as error:
            errors[variant] = f"{type(error).__name__}: {error}"
            print(f"{variant} comparison stopped: {errors[variant]}", flush=True)
    if not completed:
        (output / "errors.json").write_text(json.dumps(errors, indent=2))
        raise ValueError(f"No pose variant completed; inspect {output / 'errors.json'}")
    winner = max(completed, key=lambda r: validation_key(r["validation"]))
    comparison = {"selection_uses": "validation only; test scores never choose the extractor or classifier",
        "selected_variant": winner["variant"], "candidates": completed, "errors": errors,
        "limitations": "Exploratory reused pilot data. No unseen-infant, medical, Pi-speed, or 95% claim. Colab timings are not Pi timings."}
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\nSelected by validation: {winner['variant']} / {winner['validation']['selected_classifier']}", flush=True)
    model_dir = output / "best-model"
    if (model_dir / "test.json").exists():
        report = json.loads((model_dir / "test.json").read_text())
        if report.get("features_sha256") != _sha(winner["features"]) or report.get("model_sha256") != _sha(model_dir / "posture.joblib"):
            raise ValueError("Saved final model/report no longer match")
    else:
        report = train(winner["features"], model_dir, seed, exploratory=True, search=True)
    print(f"Final test accuracy (unknown counts as error): {report['accuracy_all_images_unknown_counts_as_error']:.1%}", flush=True)
    print("Final test rejection reasons:", report["rejection_reasons"], flush=True)
    print("This remains exploratory; a fresh independent cohort is required for a final accuracy claim.", flush=True)
    return {"model_dir": str(model_dir), "features": winner["features"],
            "comparison_path": str(output / "comparison.json"), "report": report}
