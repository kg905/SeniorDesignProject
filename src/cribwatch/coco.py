"""COCO posture intake, annotation provenance, and conservative duplicate groups.

Imported publisher labels are NOT marked as independently human-verified here.
Source-image groups prevent known duplicate leakage, not same-baby leakage.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from .data import MANIFEST_COLUMNS, write_csv

POSTURES = {"supine": "back", "back": "back", "side": "side", "prone": "prone"}


class UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, x):
        while x != self.parents[x]:
            self.parents[x] = self.parents[self.parents[x]]
            x = self.parents[x]
        return x

    def join(self, a, b):
        self.parents[self.find(b)] = self.find(a)


def perceptual_hashes(image):
    """Eight rotation/reflection variants, excluding DC from the hash threshold."""
    image = ImageOps.exif_transpose(image).convert("L").resize((32, 32))
    a = np.asarray(image, dtype=float)
    result = []
    for flipped in (a, np.fliplr(a)):
        for k in range(4):
            low = dctn(np.rot90(flipped, k), norm="ortho")[:8, :8].ravel()
            bits = low > np.median(low[1:])
            bits[0] = False
            result.append(sum(int(bit) << i for i, bit in enumerate(bits)))
    return result


def audit_coco(roots, output, near_distance=4):
    """roots maps stable dataset names to extracted export folders."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new audit directory to preserve prior results")
    rows, hashes, metadata = [], [], []
    for dataset, root in sorted(roots.items()):
        root = Path(root).resolve()
        annotation_paths = sorted(root.rglob("_annotations.coco.json"))
        if not annotation_paths:
            raise ValueError(f"No COCO annotations in {dataset}")
        for annotation_path in annotation_paths:
            obj = json.loads(annotation_path.read_text())
            categories = {c["id"]: c["name"].lower() for c in obj["categories"]}
            by_image = defaultdict(list)
            image_ids = {image["id"] for image in obj["images"]}
            if len(image_ids) != len(obj["images"]):
                raise ValueError("Duplicate COCO image IDs within a split")
            for ann in obj["annotations"]:
                if ann["image_id"] not in image_ids or ann["category_id"] not in categories:
                    raise ValueError("COCO annotation references an absent image/category")
                by_image[ann["image_id"]].append(ann)
            metadata.append({"dataset": dataset, "split": annotation_path.parent.name,
                             "images": len(image_ids), "categories": list(categories.values()),
                             "declared_licenses": obj.get("licenses", [])})
            for image in obj["images"]:
                path = (annotation_path.parent / image["file_name"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("COCO image path escapes its dataset root")
                annotations = by_image[image["id"]]
                posture_annotations = [a for a in annotations if categories[a["category_id"]] in POSTURES]
                labels = [POSTURES[categories[a["category_id"]]] for a in posture_annotations]
                known = len(labels) == 1 and not posture_annotations[0].get("iscrowd", 0)
                posture = labels[0] if known else "unknown"
                data = path.read_bytes()
                with Image.open(path) as im:
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    hashes.append(perceptual_hashes(rgb))
                source = str(image.get("extra", {}).get("name") or re.sub(r"\.rf\..*$", "", path.name))
                rows.append({
                    "image_path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                    "baby_id": "", "session_id": "", "posture": posture,
                    "face_state": "unknown", "reviewed": "0", "human_verified": "0",
                    "dataset": dataset, "original_split": annotation_path.parent.name,
                    "source_name": source, "pixel_sha256": pixel_hash,
                    "annotation_labels": "|".join(categories[a["category_id"]] for a in annotations),
                    "label_source": "dataset_annotation" if known else "none",
                    "coco_annotation_count": str(len(annotations)),
                    "posture_annotation_count": str(len(labels)),
                    "annotation_eligible": str(int(known)),
                })
    if not rows:
        raise ValueError("No COCO images")
    uf, original_uf = UnionFind(len(rows)), UnionFind(len(rows))
    link_counts = Counter()
    for field in ("sha256", "pixel_sha256", "source_name"):
        first = {}
        for i, row in enumerate(rows):
            # A generic original filename can refer to unrelated images in two
            # projects. Names alone only identify a source within one export.
            key = (row["dataset"], row[field].casefold()) if field == "source_name" else row[field].casefold()
            if key in first:
                uf.join(first[key], i)
                original_uf.join(first[key], i)
                link_counts[field] += 1
            else:
                first[key] = i
    array = np.asarray(hashes, dtype=np.uint64)
    # Compare one orientation against all eight orientations of the other image.
    # This is O(N^2), vectorized; suitable for these ~6k images, not web-scale data.
    near_pairs = []
    for i in range(len(rows)):
        distances = np.bitwise_count(array[i + 1:] ^ array[i, 0]).min(axis=1) if i + 1 < len(rows) else []
        for offset in np.flatnonzero(np.asarray(distances) <= near_distance):
            j = i + 1 + int(offset)
            if uf.find(i) != uf.find(j):
                near_pairs.append({"a": i, "b": j, "distance": int(distances[offset])})
                uf.join(i, j)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    originals = defaultdict(list)
    for i in range(len(rows)):
        originals[original_uf.find(i)].append(i)
    conflicts, cross_split, transferred = [], [], 0
    # Similar frames may depict legitimate posture changes. Only exact pixels,
    # exact files, or a declared same-source augmentation can conflict in labels.
    for members in originals.values():
        group = "original-" + hashlib.sha256("|".join(sorted({rows[i]["sha256"] for i in members})).encode()).hexdigest()[:20]
        labels = {rows[i]["posture"] for i in members if rows[i]["annotation_eligible"] == "1"}
        conflict = len(labels) > 1
        if conflict:
            conflicts.append({"group": group, "labels": sorted(labels), "row_indices": members})
        for i in members:
            row = rows[i]
            row["original_image_group"] = group
            row["label_conflict"] = str(int(conflict))
            row["training_eligible"] = str(int(row["annotation_eligible"] == "1" and not conflict))
            row["suggested_posture"] = "unknown"
            # A duplicate-group match is a suggestion, never automatic ground truth.
            if row["posture"] == "unknown" and len(labels) == 1:
                row["suggested_posture"] = next(iter(labels))
                row["suggestion_source"] = "duplicate_group_annotation"
                transferred += 1
            else:
                row["suggestion_source"] = "none"
    for members in components.values():
        group = "source-" + hashlib.sha256("|".join(sorted({rows[i]["sha256"] for i in members})).encode()).hexdigest()[:20]
        if len({rows[i]["original_split"] for i in members}) > 1:
            cross_split.append(group)
        for i in members:
            rows[i]["source_group"] = group
    summary = {
        "images": len(rows), "datasets": dict(Counter(r["dataset"] for r in rows)),
        "metadata": metadata, "posture_annotation_counts": dict(Counter(r["posture"] for r in rows)),
        "eligible_posture_images": sum(r["training_eligible"] == "1" for r in rows),
        "source_image_groups": len(components), "declared_original_image_groups": len(originals),
        "groups_crossing_original_splits": len(cross_split),
        "conflicting_label_groups": len(conflicts), "conflicting_label_images": sum(len(c["row_indices"]) for c in conflicts),
        "exact_unique_files": len({r["sha256"] for r in rows}),
        "duplicate_links": dict(link_counts), "near_duplicate_group_links": len(near_pairs),
        "near_duplicate_max_hamming": near_distance, "unknown_with_duplicate_label_suggestion": transferred,
        "baby_identity_available": False, "recording_identity_available": False,
        "limitation": "Source-image groups are conservative duplicate candidates, not infant identities. Publisher labels require independent review. This audit does not certify licenses, labels, posture accuracy, or airway outcomes.",
    }
    output.mkdir(parents=True, exist_ok=True)
    columns = MANIFEST_COLUMNS + [k for k in rows[0] if k not in MANIFEST_COLUMNS]
    write_csv(output / "images.csv", rows, columns)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "label_conflicts.json").write_text(json.dumps(conflicts, indent=2) + "\n")
    (output / "near_duplicate_links.json").write_text(json.dumps(near_pairs, indent=2) + "\n")
    return summary
