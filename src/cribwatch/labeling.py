"""Architecture-independent review/export of the user's original COCO datasets."""

from collections import Counter
import hashlib
from pathlib import Path
import shutil

from .data import read_csv, write_csv


def resolve_image(value, csv_path, root):
    path = Path(value)
    candidates = [path, Path(csv_path).parent / path, Path(root) / path]
    if "data/datasets/" in value:
        candidates.append(Path(root) / "data/datasets" / value.split("data/datasets/", 1)[1])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve {value}; set root to the checkout")


def prepare_review(source, output, *, root="."):
    if Path(output).exists():
        raise FileExistsError("Keep the existing review CSV; it stores your work")
    rows = read_csv(source)
    for row in rows:
        row["image_path"] = str(resolve_image(row["image_path"], source, root))
        row.setdefault("reviewed", "0")
        row.setdefault("human_verified", "0")
        row.setdefault("baby_id", "")
        row.setdefault("session_id", "")
        row.setdefault("face_state", "unknown")
        row.setdefault("label_source", "none")
        row["review_priority"] = "0" if row.get("label_conflict") == "1" else "1" if row.get("posture") == "unknown" else "2"
    rows.sort(key=lambda r: (r["review_priority"], r.get("source_group", ""), r["sha256"]))
    write_csv(output, rows, list(rows[0]))
    return dict(Counter(r["review_priority"] for r in rows))


def export_reviewed(manifest, output, *, folders=False):
    """Export human-confirmed labels only; optional folders are copies, not train/test splits."""
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new export folder")
    rows = [r for r in read_csv(manifest) if r.get("reviewed") == "1" and r.get("label_source") == "human_review"
            and r.get("posture") in {"back", "side", "prone", "no_baby"} and r.get("label_conflict", "0") != "1"]
    if not rows:
        raise ValueError("No reviewed non-conflicting labels yet; unclear images remain excluded")
    seen, selected = {}, []
    for row in rows:
        digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError("Image changed after review")
        if digest in seen and seen[digest] != row["posture"]:
            raise ValueError("Two reviews disagree on an identical image; adjudicate before export")
        if digest not in seen:
            selected.append(dict(row))
        seen[digest] = row["posture"]
    out.mkdir(parents=True)
    if folders:
        for row in selected:
            source = Path(row["image_path"])
            target = out / row["posture"] / (row["sha256"] + source.suffix.lower())
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            row["image_path"] = str(target.resolve())
    write_csv(out / "reviewed.csv", selected, list(selected[0]))
    return dict(Counter(r["posture"] for r in selected))


def resolve_reviewed_conflicts(manifest):
    """Clear publisher conflicts only for completely reviewed, consistent original groups."""
    rows, groups = read_csv(manifest), {}
    for row in rows:
        group = row.get("original_image_group", "")
        if group:
            groups.setdefault(group, []).append(row)
    resolved = 0
    for group in groups.values():
        if not any(r.get("label_conflict") == "1" for r in group):
            continue
        if (all(r.get("reviewed") == "1" and r.get("label_source") == "human_review" for r in group)
                and len({r["posture"] for r in group}) == 1
                and group[0]["posture"] in {"back", "side", "prone", "no_baby"}):
            for row in group:
                row.update(label_conflict="0", training_eligible="1")
            resolved += 1
    write_csv(manifest, rows, list(rows[0]))
    return {"resolved_original_groups": resolved, "remaining_flagged_images": sum(r.get("label_conflict") == "1" for r in rows)}


def rebase_reviewed(manifest, reference_index, output, *, root="."):
    """Match shared reviewed CSVs to original uploads by SHA256, not Colab paths."""
    if Path(output).exists():
        raise FileExistsError("Preserving the existing rebased CSV")
    index = {r["sha256"]: r for r in read_csv(reference_index)}
    rows = read_csv(manifest)
    for row in rows:
        if row.get("sha256") in index:
            path = resolve_image(index[row["sha256"]]["image_path"], reference_index, root)
        else:
            path = resolve_image(row["image_path"], manifest, root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Reviewed image checksum mismatch")
        row["image_path"] = str(path)
    if not rows:
        raise ValueError("No reviewed rows supplied")
    write_csv(output, rows, list(rows[0]))
    return len(rows)
