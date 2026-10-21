"""Explicit manifests, exact-content deduplication, and batched notebook review."""

import csv
import hashlib
from pathlib import Path
import math

from PIL import Image, ImageOps

from .features import FEATURES, SCHEMA_VERSION, VISION_PREPROCESSING

MANIFEST_COLUMNS = ["image_path", "sha256", "baby_id", "session_id", "posture", "face_state", "reviewed"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def make_manifest(root, output):
    if Path(output).exists():
        raise FileExistsError("Preserving existing labels; use a new manifest filename")
    rows, seen = [], set()
    for path in sorted(Path(root).resolve().rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(path) as im:
                im.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(dict(image_path=str(path), sha256=digest, baby_id="", session_id="",
                             posture="unknown", face_state="unknown", reviewed="0"))
    if not rows:
        raise ValueError("No readable images found")
    write_csv(output, rows, MANIFEST_COLUMNS)
    return len(rows)


def review_widget(manifest_path, batch_size=12):
    """Batch review avoids moving files; never asserts unreviewed labels as ground truth."""
    import ipywidgets as w
    from IPython.display import display, clear_output
    import io
    import html

    rows = read_csv(manifest_path)
    queue = [i for i, r in enumerate(rows) if r["reviewed"] != "1"]
    position = [0]
    out = w.Output()

    def render():
        with out:
            clear_output(wait=True)
            batch = queue[position[0]:position[0] + batch_size]
            if not batch:
                print("Review complete. Inspect labels before training.")
                return
            controls = []
            for i in batch:
                row = rows[i]
                with Image.open(row["image_path"]) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((260, 190))
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG")
                posture = w.Dropdown(options=["unknown", "back", "side", "prone"], value=row["posture"], description="Posture")
                face = w.Dropdown(options=["unknown", "visible", "turned_away", "occluded"], value=row["face_state"], description="Face")
                baby = w.Text(value=row["baby_id"], description="Baby ID")
                session = w.Text(value=row["session_id"], description="Session")
                controls.append((i, posture, face, baby, session))
                hint = html.escape(row.get("suggested_posture", "none"))
                display(w.VBox([w.HTML(value=f"Image {i + 1}; model suggestion: {hint} (verify visually)"),
                                w.Image(value=buf.getvalue()), posture, face, baby, session]))
            button = w.Button(description="Save reviewed batch", button_style="success")
            status = w.HTML()

            def save(_):
                # A reviewer can verify posture without knowing identity. The
                # independent training path still requires actual baby/session IDs.
                for i, p, f, b, s in controls:
                    rows[i].update(posture=p.value, face_state=f.value, baby_id=b.value.strip(),
                                   session_id=s.value.strip(), reviewed="1")
                    if "human_verified" in rows[i]:
                        rows[i]["human_verified"] = "1"
                        rows[i]["label_source"] = "human_review"
                    if "training_eligible" in rows[i]:
                        rows[i]["training_eligible"] = str(int(p.value != "unknown" and rows[i].get("label_conflict", "0") != "1"))
                write_csv(manifest_path, rows, list(rows[0]))
                position[0] += len(batch)
                render()
            button.on_click(save)
            display(button, status)
    render()
    display(out)
    return out


_worker_vision = None


def _init_extractor(model_dir):
    from multiprocessing.util import Finalize
    from .vision import Vision
    global _worker_vision
    _worker_vision = Vision(model_dir)
    # Multiprocessing finalizers run before the native dispatcher's thread pool
    # shuts down; normal Python atexit callbacks run too late for MediaPipe.
    Finalize(None, _worker_vision.close, exitpriority=10)


def _extract_row(row):
    import numpy as np
    path = Path(row["image_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"Image changed since labeling: {path}")
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    observation = _worker_vision.observe(rgb)
    return {"image_path": row["image_path"], "sha256": row["sha256"],
            "metrics": {k: v if math.isfinite(v) else None for k, v in observation.metrics.items()},
            "pose_ok": int(observation.pose_ok), "quality_reason": observation.reason,
            "preprocessing": VISION_PREPROCESSING, "feature_schema": SCHEMA_VERSION}


def extract_manifest(manifest, output, model_dir, *, workers=1):
    """Preserve annotation provenance; checkpoint expensive native inference."""
    import json
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    rows = read_csv(manifest)
    if not rows:
        raise ValueError("Empty manifest")
    if Path(output).exists():
        raise FileExistsError("Use a new feature output path")
    if not 1 <= workers <= 4:
        raise ValueError("Choose 1–4 extraction workers")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(output.suffix + ".checkpoint.jsonl")
    header = {"manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "schema": SCHEMA_VERSION, "preprocessing": VISION_PREPROCESSING,
              "models": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(model_dir).glob("*.task"))}}
    cached = {}
    if checkpoint.exists():
        with checkpoint.open() as f:
            if json.loads(next(f)) != header:
                raise ValueError("Checkpoint input/model/schema mismatch; use a new output path")
            for line in f:
                record = json.loads(line)
                cached[(record["image_path"], record["sha256"])] = record
    else:
        checkpoint.write_text(json.dumps(header) + "\n")
    todo = [r for r in rows if (r["image_path"], r["sha256"]) not in cached]
    for row in rows:
        if (row["image_path"], row["sha256"]) in cached and hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("A checkpointed image changed; preserve the old run and create a new manifest")

    def consume(results):
        with checkpoint.open("a") as f:
            for result in results:
                f.write(json.dumps(result, allow_nan=False) + "\n")
                f.flush()
                cached[(result["image_path"], result["sha256"])] = result
                if len(cached) % 200 == 0:
                    print(f"Extracted {len(cached)}/{len(rows)} images", flush=True)
    if todo:
        if workers == 1:
            _init_extractor(str(model_dir))
            try:
                consume(map(_extract_row, todo))
            finally:
                _worker_vision.close()
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init_extractor, initargs=(str(model_dir),)) as pool:
                consume(pool.map(_extract_row, todo, chunksize=8))
    for row in rows:
        record = cached[(row["image_path"], row["sha256"])]
        row.update({k: math.nan if v is None else v for k, v in record["metrics"].items()})
        row.update({k: record[k] for k in ("pose_ok", "quality_reason", "preprocessing", "feature_schema")})
    columns = list(rows[0])
    write_csv(output, rows, columns)
    return len(rows)
