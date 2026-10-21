import hashlib
import json
import zipfile

import pytest

from cribwatch.transfer import reassemble


def fixture_parts(tmp_path):
    original = bytes(range(256)) * 31
    chunks = [original[:4000], original[4000:]]
    manifest = {"format": "cribwatch-zip-parts-v1", "file_name": "original.zip", "size": len(original), "parts": []}
    for i, chunk in enumerate(chunks):
        name = f"part-{i}.zip"
        with zipfile.ZipFile(tmp_path / name, "w") as z:
            z.writestr("chunk.bin", chunk)
        manifest["parts"].append({"name": name, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, original, manifest


def test_byte_exact_reconstruction_and_existing_output_protection(tmp_path):
    manifest, original, _ = fixture_parts(tmp_path)
    output = tmp_path / "restored.zip"
    result = reassemble(manifest, tmp_path, output)
    assert output.read_bytes() == original
    assert result["sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(FileExistsError):
        reassemble(manifest, tmp_path, output)


def test_corrupt_chunk_is_not_published(tmp_path):
    manifest, original, _ = fixture_parts(tmp_path)
    with zipfile.ZipFile(tmp_path / "part-0.zip", "w") as z:
        z.writestr("chunk.bin", b"x" * 4000)
    output = tmp_path / "restored.zip"
    with pytest.raises(ValueError, match="checksum"):
        reassemble(manifest, tmp_path, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".reassemble-*"))


def test_missing_part_and_path_escape_are_rejected(tmp_path):
    manifest, _, info = fixture_parts(tmp_path)
    (tmp_path / "part-0.zip").unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        reassemble(manifest, tmp_path, tmp_path / "restored.zip")
    info["parts"][0]["name"] = "../outside.zip"
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Unsafe"):
        reassemble(manifest, tmp_path, tmp_path / "restored.zip")


def test_incomplete_manifest_cannot_publish_truncated_archive(tmp_path):
    manifest, _, info = fixture_parts(tmp_path)
    info["parts"].pop()
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="sizes"):
        reassemble(manifest, tmp_path, tmp_path / "restored.zip")
