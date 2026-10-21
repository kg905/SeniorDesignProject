"""Reconstruct user-supplied ZIP chunks after transfer-size-limited uploads."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile


def reassemble(manifest, parts_dir, output):
    """Verify every chunk before publishing a byte-exact reconstructed archive.

    The browser helper emits ordinary ZIP containers with a single chunk.bin
    entry, and a manifest with SHA256 of each original-byte chunk. No archive
    content is executed or extracted by this operation.
    """
    parts_dir, output = Path(parts_dir).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Preserving existing output; choose a new path")
    if Path(manifest).stat().st_size > 2_000_000:
        raise ValueError("Manifest is unexpectedly large")
    info = json.loads(Path(manifest).read_text())
    if not isinstance(info, dict) or info.get("format") != "cribwatch-zip-parts-v1":
        raise ValueError("Unrecognized chunk manifest format")
    size, parts = info.get("size"), info.get("parts")
    if type(size) is not int or not 0 < size <= 100_000_000_000:
        raise ValueError("Invalid original archive size")
    if not isinstance(parts, list) or not 0 < len(parts) <= 10_000:
        raise ValueError("Invalid part list")
    names, total = set(), 0
    for part in parts:
        if not isinstance(part, dict):
            raise ValueError("Invalid part description")
        name, count, digest = part.get("name"), part.get("size"), part.get("sha256")
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Unsafe part filename")
        if name in names:
            raise ValueError("Duplicate part filename")
        names.add(name)
        if type(count) is not int or not 0 < count <= 24 * 1024 * 1024:
            raise ValueError("Invalid chunk size")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Invalid chunk SHA256")
        path = (parts_dir / name).resolve()
        if path.parent != parts_dir:
            raise ValueError("Part resolves outside the supplied directory")
        if not path.is_file():
            raise FileNotFoundError(f"Missing part: {name}")
        total += count
    if total != size:
        raise ValueError("Manifest part sizes do not match original size")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    full_hash = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".reassemble-", delete=False) as destination:
            temporary = Path(destination.name)
            for part in parts:
                with zipfile.ZipFile(parts_dir / part["name"]) as z:
                    entries = z.infolist()
                    if len(entries) != 1 or entries[0].filename != "chunk.bin" or entries[0].file_size != part["size"]:
                        raise ValueError("Unexpected chunk ZIP contents/size")
                    # ZipFile also verifies each entry's CRC when reading it.
                    chunk = z.read(entries[0])
                if hashlib.sha256(chunk).hexdigest() != part["sha256"]:
                    raise ValueError(f"Chunk checksum mismatch: {part['name']}")
                destination.write(chunk)
                full_hash.update(chunk)
        # Atomic, and refuses to overwrite a file created since the initial check.
        os.link(temporary, output)
        return {"bytes": size, "parts": len(parts), "sha256": full_hash.hexdigest()}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
