import hashlib
import io

import pytest

from cribwatch import assets


def configure(monkeypatch, payload=b"verified model"):
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(assets, "ASSETS", {"pose.task": ("https://example.test/pose", digest)})
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(payload))
    return digest


def test_selected_model_replaces_matching_lfs_pointer(tmp_path, monkeypatch):
    digest = configure(monkeypatch)
    target = tmp_path / "pose.task"
    target.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize 14\n")
    assets.download_models(tmp_path, names=["pose.task"])
    assert target.read_bytes() == b"verified model"


def test_invalid_existing_model_is_preserved(tmp_path, monkeypatch):
    configure(monkeypatch)
    target = tmp_path / "pose.task"
    target.write_bytes(b"unexpected existing model")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        assets.download_models(tmp_path, names=["pose.task"])
    assert target.read_bytes() == b"unexpected existing model"


def test_failed_verification_preserves_pointer(tmp_path, monkeypatch):
    digest = configure(monkeypatch)
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"bad download"))
    target = tmp_path / "pose.task"
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize 14\n"
    target.write_text(pointer)
    with pytest.raises(ValueError, match="download checksum mismatch"):
        assets.download_models(tmp_path, names=["pose.task"])
    assert target.read_text() == pointer
    assert not target.with_suffix(".download").exists()
