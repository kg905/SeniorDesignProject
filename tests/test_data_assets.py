import zipfile

from PIL import Image
import pytest

from cribwatch.assets import safe_unzip
from cribwatch.data import make_manifest, read_csv


def test_manifest_deduplicates_without_inventing_labels(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (20, 20), "blue").save(source / "a.png")
    (source / "b.png").write_bytes((source / "a.png").read_bytes())
    manifest = tmp_path / "manifest.csv"
    assert make_manifest(source, manifest) == 1
    row = read_csv(manifest)[0]
    assert row["posture"] == "unknown" and row["reviewed"] == "0" and row["baby_id"] == ""
    with pytest.raises(FileExistsError):
        make_manifest(source, manifest)


def test_zip_traversal_refused(tmp_path):
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escape", "bad")
    with pytest.raises(ValueError):
        safe_unzip(path, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()
