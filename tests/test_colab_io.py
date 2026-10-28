import io
from pathlib import Path
import zipfile

import pytest

from cribwatch.colab_io import local_zip_run


def archive_bytes():
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("BabyTraining/back/example.txt", "upload plumbing fixture")
    return data.getvalue()


def test_upload_writes_to_local_run_and_restores_working_directory(tmp_path):
    original = Path.cwd()
    data = archive_bytes()
    def uploader():
        assert Path.cwd().is_relative_to(tmp_path / "local")
        Path("BabyTraining.zip").write_bytes(data)
        return {"BabyTraining.zip": data}
    run, path = local_zip_run(work_root=tmp_path / "local", uploader=uploader)
    assert Path.cwd() == original and path.parent == run / "incoming"
    assert path.read_bytes() == data and (run / "input.json").is_file()


def test_lost_previous_mount_does_not_break_completed_local_upload(tmp_path, monkeypatch):
    previous = tmp_path / "old-drive"
    previous.mkdir()
    monkeypatch.chdir(previous)
    def uploader():
        previous.rmdir()  # Simulate a mount/cwd disappearing during the transfer.
        return {"BabyTraining.zip": archive_bytes()}
    run, path = local_zip_run(work_root=tmp_path / "local", uploader=uploader)
    assert path.is_file() and Path.cwd() == run


def test_reading_existing_zip_copies_it_locally_without_changing_original(tmp_path):
    source = tmp_path / "source.zip"
    source.write_bytes(archive_bytes())
    run, path = local_zip_run(work_root=tmp_path / "local", source_zip=source)
    assert path != source and path.read_bytes() == source.read_bytes()
    assert path.is_relative_to(run)


def test_incomplete_or_invalid_upload_is_not_accepted(tmp_path):
    def incomplete():
        Path("BabyTraining.zip").write_bytes(b"partial")
        return {"BabyTraining.zip": archive_bytes()}
    with pytest.raises(ValueError, match="incomplete"):
        local_zip_run(work_root=tmp_path / "local", uploader=incomplete)
    with pytest.raises(ValueError, match="valid ZIP"):
        local_zip_run(work_root=tmp_path / "local", uploader=lambda: {"image.jpg": b"not a zip"})


def test_drive_work_directory_is_rejected_before_upload():
    with pytest.raises(ValueError, match="local"):
        local_zip_run(work_root="/content/drive/MyDrive/run", uploader=lambda: pytest.fail("Must not upload to Drive"))
