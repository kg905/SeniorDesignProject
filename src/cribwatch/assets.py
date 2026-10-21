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


def download_models(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in ASSETS.items():
        target = directory / name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Checksum mismatch for existing {target}; preserve/investigate or remove it before downloading")
            continue
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
