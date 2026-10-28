"""Keep uploads and active runs off Google Drive's disconnectable FUSE mount."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4
import zipfile
from zoneinfo import ZoneInfo


def local_zip_run(*, work_root="/content/cribwatch-runs", uploader=None, source_zip=None):
    """Return local (RUN, ZIP_PATH); optionally read an already-mounted source ZIP."""
    work = Path(work_root).absolute()
    if work == Path("/content/drive") or Path("/content/drive") in work.parents:
        raise ValueError("Use local /content storage for work_root, not mounted Google Drive")
    work.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M%S")
    run = work / ("run-" + stamp + "-" + uuid4().hex[:6])
    incoming = run / "incoming"
    incoming.mkdir(parents=True)
    if source_zip:
        source = Path(source_zip)
        try:
            if not source.is_file():
                raise FileNotFoundError(source)
            target = incoming / source.name
            shutil.copyfile(source, target)
        except OSError as error:
            raise OSError("Cannot read the source ZIP. If Drive disconnected, leave ZIP_IN_DRIVE empty and upload from your computer.") from error
    else:
        if uploader is None:
            from google.colab import files
            uploader = files.upload
        try:
            previous = Path.cwd()
        except OSError:
            previous = work
        try:
            os.chdir(incoming)
            uploaded = uploader()
        finally:
            try:
                os.chdir(previous)
            except OSError:
                os.chdir(run)
        if len(uploaded) != 1:
            raise ValueError("Select ONE sorted BabyTraining.zip, not individual photos or multiple ZIPs")
        name, data = next(iter(uploaded.items()))
        target = incoming / Path(name).name
        if not target.is_file():
            target.write_bytes(data)
        # A successful browser transfer is not sufficient if the written file is incomplete.
        if target.stat().st_size != len(data):
            raise ValueError("Uploaded ZIP is incomplete; rerun this cell and upload the same ZIP again")
    if not zipfile.is_zipfile(target):
        raise ValueError("Select a valid ZIP containing back/, side/, and prone/ folders")
    with zipfile.ZipFile(target) as archive:
        members = archive.infolist()
        if len(members) > 100_000 or sum(m.file_size for m in members) > 4_000_000_000:
            raise ValueError("ZIP exceeds the supported file-count/uncompressed-size limits")
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed; upload the original sorted ZIP again")
    (run / "input.json").write_text(json.dumps({"zip_name": target.name,
        "zip_bytes": target.stat().st_size, "zip_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "storage": "local_colab_download_results_before_runtime_reset"}, indent=2) + "\n")
    return run, target
