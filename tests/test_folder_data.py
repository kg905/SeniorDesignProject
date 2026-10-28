import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from cribwatch.assets import safe_unzip
from cribwatch.data import read_csv, write_csv
from cribwatch.folder_data import import_folders


def folders(tmp_path):
    root = tmp_path / "BabyTraining"
    for name in ["back", "side", "prone", "unknown"]:
        (root / name).mkdir(parents=True)
    return root


def picture(path, seed):
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(path)
    return path


def test_manual_folder_labels_override_filename_and_publisher_label(tmp_path):
    root = folders(tmp_path)
    photo = picture(root / "back/prone_in_original_filename.png", 1)
    picture(root / "side/a.png", 2)
    picture(root / "prone/a.png", 3)
    picture(root / "unknown/a.png", 4)
    reference = tmp_path / "original.csv"
    write_csv(reference, [dict(sha256=hashlib.sha256(photo.read_bytes()).hexdigest(), posture="prone",
                              source_group="known-source", original_image_group="known-original")],
              ["sha256", "posture", "source_group", "original_image_group"])
    summary = import_folders(root, tmp_path / "output", reference_index=reference)
    rows = read_csv(tmp_path / "output/images.csv")
    assert summary["matched_original_images"] == 1
    assert summary["eligible_unique_counts"] == {"back": 1, "prone": 1, "side": 1}
    row = next(r for r in rows if r["posture"] == "back")
    assert row["reviewed"] == row["human_verified"] == "1"
    assert row["label_source"] == "human_review" and row["baby_id"] == ""
    assert next(r for r in rows if r["posture"] == "unknown")["training_eligible"] == "0"
    assert photo.exists()  # Source files are never moved.


def test_exact_duplicate_in_different_classes_is_quarantined(tmp_path):
    root = folders(tmp_path)
    a = picture(root / "back/a.png", 1)
    (root / "prone/renamed.png").write_bytes(a.read_bytes())
    summary = import_folders(root, tmp_path / "output")
    assert summary["exact_duplicate_files"] == 1
    assert summary["conflicting_images"] == 2
    assert all(r["training_eligible"] == "0" for r in read_csv(tmp_path / "output/images.csv"))
    assert len(json.loads((tmp_path / "output/conflicts.json").read_text())) == 1


def test_known_related_sources_stay_together_without_confusing_posture_change(tmp_path):
    root = folders(tmp_path)
    images = [picture(root / "back/a.png", 1), picture(root / "side/b.png", 2), picture(root / "prone/c.png", 3)]
    reference = tmp_path / "reference.csv"
    rows = [dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), source_group="one-recording",
                 original_image_group=f"different-frame-{i}") for i, p in enumerate(images)]
    write_csv(reference, rows, list(rows[0]))
    summary = import_folders(root, tmp_path / "out", reference_index=reference)
    imported = read_csv(tmp_path / "out/images.csv")
    assert len({r["source_group"] for r in imported}) == 1
    assert summary["conflicting_images"] == 0
    assert all(r["training_eligible"] == "1" for r in imported)


def test_zip_wrapper_and_apple_metadata_are_handled(tmp_path):
    root = folders(tmp_path)
    for i, label in enumerate(["back", "side", "prone"]):
        picture(root / label / "a.png", i)
    archive = tmp_path / "BabyTraining.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in root.rglob("*.png"):
            z.write(p, str(p.relative_to(root.parent)))
        z.writestr("__MACOSX/BabyTraining/back/._a.png", b"not an image")
    safe_unzip(archive, tmp_path / "unpacked")
    assert import_folders(tmp_path / "unpacked", tmp_path / "out")["images"] == 3


def test_optional_real_identity_metadata_groups_baby_across_classes(tmp_path):
    root = folders(tmp_path)
    for i, label in enumerate(["back", "side", "prone"]):
        picture(root / label / "a.png", i)
    ids = [dict(relative_path=f"{label}/a.png", baby_id="actual-known-child", session_id=f"session-{label}")
           for label in ["back", "side", "prone"]]
    write_csv(root / "metadata.csv", ids, list(ids[0]))
    import_folders(root, tmp_path / "out")
    rows = read_csv(tmp_path / "out/images.csv")
    assert {r["baby_id"] for r in rows} == {"actual-known-child"}
    assert len({r["source_group"] for r in rows}) == 1


def test_bad_layout_corrupt_file_and_overwrite_are_reported(tmp_path):
    with pytest.raises(ValueError, match="ONE folder"):
        import_folders(tmp_path, tmp_path / "out")
    root = folders(tmp_path)
    (root / "back/bad.jpg").write_text("not an image")
    with pytest.raises(ValueError, match="Unreadable image"):
        import_folders(root, tmp_path / "out")
    (root / "back/bad.jpg").unlink()
    picture(root / "back/a.png", 1)
    import_folders(root, tmp_path / "out")
    with pytest.raises(FileExistsError):
        import_folders(root, tmp_path / "out")
