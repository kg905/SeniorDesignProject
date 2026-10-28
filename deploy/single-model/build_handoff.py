"""Bundle only a genuinely trained AND evaluated export; never manufacture model files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build(model, output, root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    model, output = Path(model), Path(output)
    if output.exists():
        raise FileExistsError("Use a new handoff ZIP")
    report = json.loads((model / "test.json").read_text())
    if not report.get("n_images") or not report.get("model_sha256"):
        raise ValueError("Need a trained classifier and its checksum-bound evaluation report")
    if hashlib.sha256((model / "posture.joblib").read_bytes()).hexdigest() != report["model_sha256"]:
        raise ValueError("Evaluation/model checksum mismatch")
    files = {}
    files["models/posture_model.joblib"] = model / "posture.joblib"
    files["models/test.json"] = model / "test.json"
    files["models/training_feature_percentiles.json"] = model / "training_feature_percentiles.json"
    from cribwatch.assets import classifier_assets
    import joblib
    artifact = joblib.load(model / "posture.joblib")
    for name, expected in classifier_assets(artifact).items():
        path = root / "models" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("MediaPipe asset checksum mismatch")
        files[f"models/{name}"] = path
        receipt = path.with_name(path.name + ".source.json")
        if receipt.exists():
            files[f"models/{receipt.name}"] = receipt
    adapter = artifact.get("vision_adapter")
    if adapter is not None:
        from cribwatch.hybrid_vision import validate_adapter
        validate_adapter(adapter)
        path = root / "models" / adapter["yolo_name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != adapter["yolo_sha256"]:
            raise ValueError("YOLO asset checksum mismatch")
        files[f"models/{path.name}"] = path
        files["requirements-hybrid.txt"] = root / "training/requirements-hybrid.txt"
        files["HYBRID.md"] = root / "docs/HYBRID_ALTERNATIVE.md"
        for name in ("YOLO_ATTRIBUTION.md", "YOLO_LICENSE.txt"):
            if (root / "models" / name).exists():
                files[f"models/{name}"] = root / "models" / name
    for name in ("validation.json", "classifier_search.json"):
        if (model / name).exists():
            files[f"models/{name}"] = model / name
    for name in ("monitor_pi5.py", "test_model.py", "alarm_receiver.py", "requirements-pi5.txt", "cribwatch-camera.service", "cribwatch-alarm.service"):
        files[name] = root / "deploy/single-model" / name
    files["README.md"] = root / "docs/PI5_HANDOFF.md"
    files["PROTOCOL.md"] = root / "docs/ALARM_PROTOCOL.md"
    files["reference/SIDS_Pi5_Team_Handoff.pdf"] = root / "docs/reference/SIDS_Pi5_Team_Handoff.pdf"
    # Include the shared predictor/feature code, not notebooks or raw images.
    runtime_modules = ("__init__.py", "camera.py", "alarm_runtime.py", "policy.py", "transport.py",
                       "tcp_alarm.py", "display.py", "showcase.py", "metrics_model.py",
                       "learning.py", "features.py", "vision.py", "hybrid_vision.py", "data.py", "assets.py")
    for name in runtime_modules:
        path = root / "src/cribwatch" / name
        files[f"src/cribwatch/{path.name}"] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr("SHA256SUMS.json", json.dumps({name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}, indent=2))
    return str(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build(args.model, args.output))
