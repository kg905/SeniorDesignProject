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


def review_widget(manifest_path, batch_size=12, *, blind=False, show_suggestions=False):
    """Explicit widget tree; no nested Output capture that can be blank in Colab."""
    import ipywidgets as w
    from IPython.display import display
    import io
    import html
    from .review_console import save_review

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    print("Opening review controls. If no images appear, use review_console(REVIEW).", flush=True)
    container = w.VBox()
    display(container)

    def render():
        rows = read_csv(manifest_path)
        pending = [r for r in rows if r.get("reviewed") != "1"]
        progress = w.HTML(value=f"<b>Saved: {len(rows) - len(pending)} / {len(rows)}</b>")
        if not pending:
            container.children = (progress, w.HTML(value="Review complete. Continue to export."))
            return
        children, controls = [progress], []
        for n, row in enumerate(pending[:batch_size], 1):
            with Image.open(row["image_path"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((460, 340))
                buf = io.BytesIO()
                im.save(buf, format="JPEG")
            posture = w.Dropdown(options=["unknown", "back", "side", "prone", "no_baby"],
                                 value="unknown" if blind else row["posture"], description="Posture")
            face = w.Dropdown(options=["unknown", "visible", "turned_away", "occluded"],
                              value=row.get("face_state", "unknown"), description="Face")
            baby = w.Text(value=row.get("baby_id", ""), description="Baby ID")
            session = w.Text(value=row.get("session_id", ""), description="Session")
            controls.append((row, posture, face, baby, session))
            hint = f"; suggestion: {html.escape(row.get('suggested_posture', 'none'))} (verify)" if show_suggestions and not blind else ""
            group = html.escape(row.get("source_group", "unassigned"))
            conflict = "; SOURCE LABEL CONFLICT: stays quarantined until group adjudication" if row.get("label_conflict") == "1" else ""
            children.append(w.VBox([w.HTML(value=f"Image {n}; group: {group}{conflict}{hint}"),
                                    w.Image(value=buf.getvalue(), format="jpeg"), posture, face, baby, session]))
        button = w.Button(description="Save reviewed batch", button_style="success")
        status = w.HTML()

        def save(_):
            button.disabled = True
            status.value = "Saving to your review CSV..."
            try:
                for row, p, f, b, session in controls:
                    save_review(manifest_path, row, p.value, face=f.value, baby=b.value, session=session.value)
            except Exception as error:
                status.value = "<b>Save stopped:</b> " + html.escape(str(error)) + ". Rerun this cell to resume saved progress."
                return
            render()
        button.on_click(save)
        container.children = tuple(children + [button, status])
    render()
    return container


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
    print(f"Extraction: {len(cached)}/{len(rows)} images saved; {len(todo)} remaining", flush=True)
    for row in rows:
        if (row["image_path"], row["sha256"]) in cached and hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("A checkpointed image changed; preserve the old run and create a new manifest")

    def consume(results):
        with checkpoint.open("a") as f:
            for result in results:
                f.write(json.dumps(result, allow_nan=False) + "\n")
                f.flush()
                cached[(result["image_path"], result["sha256"])] = result
                if len(cached) == 1 or len(cached) % 10 == 0 or len(cached) == len(rows):
                    print(f"Extracted {len(cached)}/{len(rows)} images", flush=True)
    if todo:
        if workers == 1:
            print("Initializing MediaPipe pose and face tasks...", flush=True)
            _init_extractor(str(model_dir))
            print("MediaPipe ready; processing images...", flush=True)
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
    print(f"Saved features: {output}", flush=True)
    return len(rows)
