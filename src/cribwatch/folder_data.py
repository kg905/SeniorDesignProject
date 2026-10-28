"""Import explicit human class-folder labels; never infer posture from filenames."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps

from .coco import UnionFind, perceptual_hashes
from .data import IMAGE_SUFFIXES, read_csv, write_csv

FOLDER_LABELS = {"back": "back", "side": "side", "prone": "prone", "belly": "prone",
                 "unknown": "unknown", "no_baby": "no_baby"}
POSTURES = {"back", "side", "prone"}


def hidden(path, root):
    return any(part.startswith(".") or part == "__MACOSX" for part in path.relative_to(root).parts)


def find_class_root(root):
    """Accept either the class root itself or one enclosing ZIP folder."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("The extracted folder does not exist")
    candidates = []
    for path in [root, *sorted(p for p in root.rglob("*") if p.is_dir() and not hidden(p, root))]:
        names = {FOLDER_LABELS.get(p.name.lower()) for p in path.iterdir() if p.is_dir()}
        if POSTURES.issubset(names):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("Expected ONE folder containing back/, side/, and prone/ (or belly/). Do not create train/test subfolders.")
    return candidates[0]


def import_folders(root, output, *, reference_index=None, near_distance=4):
    """Write a reviewed manifest and audit, preserving unknowns and original copies."""
    root = find_class_root(root)
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a new output folder; existing labels/runs are preserved")
    if not 0 <= near_distance <= 8:
        raise ValueError("near_distance must be between 0 and 8")
    reference = defaultdict(list)
    if reference_index is not None:
        for row in read_csv(reference_index):
            reference[row["sha256"]].append(row)
    metadata = {}
    if (root / "metadata.csv").is_file():
        for row in read_csv(root / "metadata.csv"):
            relative = row.get("relative_path", "").replace("\\", "/")
            if not relative or relative in metadata:
                raise ValueError("metadata.csv requires unique relative_path entries, e.g. back/image.jpg")
            metadata[relative] = row
    rows, hashes, refs, ignored = [], [], [], []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        posture = FOLDER_LABELS.get(folder.name.lower())
        if posture is None:
            if not hidden(folder, root):
                ignored.append(str(folder.relative_to(root)) + "/ (unrecognized folder)")
            continue
        for path in sorted(p for p in folder.rglob("*") if p.is_file() and not hidden(p, root)):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                ignored.append(str(path.relative_to(root)))
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Image path escapes the selected folder")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    pixel_digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    hashes.append(perceptual_hashes(rgb))
            except (OSError, ValueError) as error:
                raise ValueError(f"Unreadable image: {path.relative_to(root)}") from error
            relative = path.relative_to(root).as_posix()
            identity = metadata.get(relative, {})
            matches = reference.get(digest, [])
            refs.append(matches)
            rows.append(dict(image_path=str(path.resolve()), relative_path=relative, sha256=digest,
                pixel_sha256=pixel_digest, source_name=path.name, posture=posture, face_state="unknown",
                baby_id=(identity.get("baby_id") or "").strip(), session_id=(identity.get("session_id") or "").strip(),
                reviewed="1", human_verified="1", label_source="human_review", label_method="manual_class_folder",
                folder_label=folder.name, dataset="manual_folders", source_group="", original_image_group="",
                label_conflict="0", training_eligible="0", matched_original=str(int(bool(matches)))))
    if not rows:
        raise ValueError("No supported images found; use JPG/JPEG, PNG, WEBP or BMP")
    unused = set(metadata) - {r["relative_path"] for r in rows}
    if unused:
        raise ValueError(f"metadata.csv refers to missing images: {sorted(unused)[:3]}")
    groups, originals = UnionFind(len(rows)), UnionFind(len(rows))
    owners = {}

    def join(key, i, *, original=False):
        if key in owners:
            groups.join(i, owners[key])
            if original:
                originals.join(i, owners[key])
        else:
            owners[key] = i

    for i, row in enumerate(rows):
        join(("sha", row["sha256"]), i, original=True)
        join(("pixels", row["pixel_sha256"]), i, original=True)
        for field in ("baby_id", "session_id"):
            if row[field]:
                join((field, row[field]), i)
        for ref in refs[i]:
            if ref.get("source_group"):
                join(("reference_source", ref["source_group"]), i)
            if ref.get("original_image_group"):
                join(("reference_original", ref["original_image_group"]), i, original=True)
        # A Roboflow .rf. suffix declares derived versions of the same named image.
        # This only groups unknown-source copies; it never supplies posture labels.
        if not refs[i] and ".rf." in row["source_name"]:
            family = re.sub(r"\.rf\..*$", "", row["source_name"], flags=re.IGNORECASE)
            join(("declared_rf_family", family.casefold()), i, original=True)
        # Optional same-named recording subfolders are source groups, never infant IDs.
        relative = Path(row["relative_path"])
        if len(relative.parts) > 2:
            join(("recording_subfolder", relative.parts[1:-1]), i)
    array = np.asarray(hashes, dtype=np.uint64)
    near_links = 0
    for i in range(len(rows) - 1):
        distances = np.bitwise_count(array[i + 1:] ^ array[i, 0]).min(axis=1)
        for offset in np.flatnonzero(distances <= near_distance):
            j = i + 1 + int(offset)
            if groups.find(i) != groups.find(j):
                groups.join(i, j)
                near_links += 1
    components, original_components = defaultdict(list), defaultdict(list)
    for i in range(len(rows)):
        components[groups.find(i)].append(i)
        original_components[originals.find(i)].append(i)
    conflicts = []
    for members in components.values():
        value = "manual-source-" + hashlib.sha256("|".join(sorted({rows[i]["sha256"] for i in members})).encode()).hexdigest()[:20]
        for i in members:
            rows[i]["source_group"] = value
    for members in original_components.values():
        value = "manual-original-" + hashlib.sha256("|".join(sorted({rows[i]["sha256"] for i in members})).encode()).hexdigest()[:20]
        labels = {rows[i]["posture"] for i in members if rows[i]["posture"] != "unknown"}
        conflict = len(labels) > 1
        if conflict:
            conflicts.append({"group": value, "labels": sorted(labels), "files": [rows[i]["relative_path"] for i in members]})
        for i in members:
            rows[i].update(original_image_group=value, label_conflict=str(int(conflict)),
                           training_eligible=str(int(rows[i]["posture"] in POSTURES and not conflict)))
    out.mkdir(parents=True)
    write_csv(out / "images.csv", rows, list(rows[0]))
    seen, counts = set(), Counter()
    for row in rows:
        if row["training_eligible"] == "1" and row["sha256"] not in seen:
            counts[row["posture"]] += 1
            seen.add(row["sha256"])
    summary = {"images": len(rows), "folder_root": str(root), "folder_counts": dict(Counter(r["posture"] for r in rows)),
        "eligible_unique_counts": dict(counts), "exact_duplicate_files": len(rows) - len({r["sha256"] for r in rows}),
        "matched_original_images": sum(bool(matches) for matches in refs), "source_groups": len(components),
        "near_duplicate_links": near_links, "conflicting_original_groups": len(conflicts),
        "conflicting_images": sum(r["label_conflict"] == "1" for r in rows), "ignored_files": ignored,
        "labels_supplied_by": "user's manual class-folder assignments; not machine predictions",
        "limitations": "Source groups are duplicate/recording candidates, not verified infant identities. Folder labels are user-supplied, not independently validated. Unknowns stay excluded from posture training. A small pilot cannot establish 95% accuracy."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "conflicts.json").write_text(json.dumps(conflicts, indent=2) + "\n")
    return summary
