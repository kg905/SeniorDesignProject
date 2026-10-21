import json
from pathlib import Path

from PIL import Image
import pytest

from cribwatch.coco import audit_coco
from cribwatch.data import read_csv


def coco_export(root, split, label, *, image_name="fixture.png", source="fixture.png"):
    folder = root / split
    folder.mkdir(parents=True)
    # Structured color pattern: meaningful pHash rather than a blank-image fixture.
    im = Image.new("RGB", (64, 64), "navy")
    for x in range(20, 44):
        for y in range(10, 50):
            im.putpixel((x, y), (x * 5, y * 4, 60))
    im.save(folder / image_name)
    data = {"categories": [{"id": 1, "name": label}],
            "images": [{"id": 0, "file_name": image_name, "extra": {"name": source}}],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 64, 64]}]}
    (folder / "_annotations.coco.json").write_text(json.dumps(data))
    return folder


def test_existing_posture_labels_keep_provenance_and_not_human_reviewed(tmp_path):
    coco_export(tmp_path / "input", "train", "supine")
    result = audit_coco({"posture": tmp_path / "input"}, tmp_path / "audit")
    row = read_csv(tmp_path / "audit/images.csv")[0]
    assert row["posture"] == "back"
    assert row["training_eligible"] == "1"
    assert row["reviewed"] == row["human_verified"] == "0"
    assert row["baby_id"] == "" and row["session_id"] == ""
    assert not result["baby_identity_available"]


def test_duplicate_conflicts_are_quarantined_and_original_split_leakage_reported(tmp_path):
    root = tmp_path / "input"
    coco_export(root, "train", "supine")
    coco_export(root, "test", "prone")
    result = audit_coco({"posture": root}, tmp_path / "audit")
    rows = read_csv(tmp_path / "audit/images.csv")
    assert result["groups_crossing_original_splits"] == 1
    assert result["conflicting_label_groups"] == 1
    assert all(r["training_eligible"] == "0" for r in rows)
    assert len({r["source_group"] for r in rows}) == 1


def test_baby_face_annotations_never_become_posture_ground_truth(tmp_path):
    coco_export(tmp_path / "input", "train", "baby_face")
    audit_coco({"baby_face": tmp_path / "input"}, tmp_path / "audit")
    row = read_csv(tmp_path / "audit/images.csv")[0]
    assert row["posture"] == "unknown" and row["training_eligible"] == "0"


def test_coco_path_escape_is_rejected(tmp_path):
    folder = coco_export(tmp_path / "input", "train", "supine")
    data = json.loads((folder / "_annotations.coco.json").read_text())
    data["images"][0]["file_name"] = "../../outside.png"
    (folder / "_annotations.coco.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        audit_coco({"input": tmp_path / "input"}, tmp_path / "audit")


def test_similar_frames_with_real_posture_changes_are_grouped_not_quarantined(tmp_path):
    root = tmp_path / "input"
    coco_export(root, "train", "supine", image_name="first.png", source="first.png")
    folder = coco_export(root, "test", "prone", image_name="second.png", source="second.png")
    with Image.open(folder / "second.png") as im:
        im.putpixel((31, 30), (155, 119, 61))
        im.save(folder / "second.png")
    result = audit_coco({"posture": root}, tmp_path / "audit")
    rows = read_csv(tmp_path / "audit/images.csv")
    assert result["source_image_groups"] == 1
    assert result["declared_original_image_groups"] == 2
    assert all(row["training_eligible"] == "1" for row in rows)
    assert all(row["label_conflict"] == "0" for row in rows)
