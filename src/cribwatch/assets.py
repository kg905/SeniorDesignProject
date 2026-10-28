"""Versioned official model assets with pinned SHA256 verification."""

import hashlib
from pathlib import Path
import urllib.request

ASSETS = {
    "mobilenet_v3_small.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite?generation=1682480013737984",
        "bbbb4c51a55a53905af1daec995ca1aae355046f8839bb8c9f5ce9271394bc40",
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task?generation=1682624736756847",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task?generation=1683136941916318",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}

# Additional official pose variants are downloaded over HTTPS and their exact
# content hashes are recorded with the trained classifier. Unlike ASSETS above,
# these hashes are recorded on first download, not pre-pinned in this repository.
POSE_VARIANTS = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

YOLO_ASSETS = {
    "yolov8n-pose.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
        "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212",
    ),
}


def download_pose_variant(directory, variant):
    import json
    import zipfile
    name = POSE_VARIANTS[variant]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    download_models(directory, names=["face_landmarker.task"])
    if name in ASSETS:
        download_models(directory, names=[name])
        return name
    url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
           f"pose_landmarker_{variant}/float16/1/{name}")
    target, receipt = directory / name, directory / (name + ".source.json")
    if target.exists():
        metadata = json.loads(receipt.read_text()) if receipt.exists() else {}
        if metadata.get("url") != url or metadata.get("sha256") != hashlib.sha256(target.read_bytes()).hexdigest():
            raise ValueError(f"Unverified existing {name}; use a fresh model directory")
        print(f"Verified recorded checksum: {name}", flush=True)
        return name
    temporary = target.with_suffix(".download")
    try:
        print(f"Downloading official MediaPipe {variant} pose asset...", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as f:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 40_000_000:
                    raise ValueError("Unexpected pose asset size")
                f.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Expected a MediaPipe task archive")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Damaged MediaPipe task archive")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(target)
        receipt.write_text(json.dumps({"url": url, "sha256": digest, "bytes": total}, indent=2))
        print(f"Recorded {name}: {total:,} bytes, SHA256 {digest}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)
    return name


def classifier_assets(artifact):
    """Old classifiers use pinned Lite; newer exports bind their exact extractor."""
    recorded = artifact.get("vision_assets")
    if recorded is None:
        return {name: ASSETS[name][1] for name in (POSE_VARIANTS["lite"], "face_landmarker.task")}
    if not isinstance(recorded, dict):
        raise ValueError("Invalid vision asset metadata")
    pose_names = set(recorded) & set(POSE_VARIANTS.values())
    if len(pose_names) != 1 or set(recorded) != pose_names | {"face_landmarker.task"}:
        raise ValueError("Expected one recognized pose asset and one face asset")
    for name, digest in recorded.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid vision asset SHA256")
        if name in ASSETS and digest != ASSETS[name][1]:
            raise ValueError("Pinned vision asset checksum mismatch")
    return recorded


def verify_classifier_assets(artifact, directory):
    recorded = classifier_assets(artifact)
    for name, expected in recorded.items():
        if hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest() != expected:
            raise ValueError("MediaPipe asset checksum mismatch")
    return next(name for name in recorded if name != "face_landmarker.task")


def download_models(directory, *, names=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ASSETS if names is None else names:
        url, digest = (ASSETS | YOLO_ASSETS)[name]
        target = directory / name
        if target.exists():
            existing = target.read_bytes()
            if hashlib.sha256(existing).hexdigest() == digest:
                print(f"Verified {name}", flush=True)
                continue
            is_pointer = (existing.startswith(b"version https://git-lfs.github.com/spec/v1\n")
                          and f"oid sha256:{digest}\n".encode() in existing)
            if not is_pointer:
                raise ValueError(f"Checksum mismatch for existing {target}; preserve/investigate or remove it before downloading")
        print(f"Downloading {name} from official MediaPipe storage...", flush=True)
        temporary = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as f:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 20_000_000:
                        raise ValueError("Unexpected model size")
                    f.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Model download checksum mismatch: {name}")
            temporary.replace(target)
            print(f"Verified {name}: {total:,} bytes", flush=True)
        finally:
            temporary.unlink(missing_ok=True)


def safe_unzip(archive, destination, max_bytes=4_000_000_000, max_files=100_000):
    """Extract into a new directory, with traversal, symlink and size checks."""
    import zipfile
    import stat
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Extract into a new directory to preserve existing data")
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if len(members) > max_files or sum(m.file_size for m in members) > max_bytes:
            raise ValueError("Archive exceeds configured extraction limits")
        for m in members:
            target = (destination / m.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(m.external_attr >> 16):
                raise ValueError("Unsafe archive path or symlink")
        destination.mkdir(parents=True)
        z.extractall(destination)
