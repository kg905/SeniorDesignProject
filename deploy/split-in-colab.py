# Paste this cell into a Google Colab notebook and run it.
# Upload the original Roboflow ZIPs, then use Colab's Files panel to download
# every generated part ZIP and manifest from the printed output folder.
from google.colab import files
from pathlib import Path
import hashlib
import json
import uuid
import zipfile

uploaded = files.upload()
if not uploaded:
    raise ValueError("Choose at least one original dataset ZIP")
output = Path("cribwatch_parts_" + uuid.uuid4().hex[:8])
output.mkdir()
chunk_size = 24 * 1024 * 1024
for original_name, data in uploaded.items():
    name = Path(original_name).name
    if not name.lower().endswith(".zip") or not data:
        raise ValueError("Choose nonempty original dataset ZIP files")
    manifest = {"format": "cribwatch-zip-parts-v1", "file_name": name, "size": len(data), "parts": []}
    count = (len(data) + chunk_size - 1) // chunk_size
    for i in range(count):
        chunk = data[i * chunk_size:(i + 1) * chunk_size]
        part = f"{name}.part-{i + 1:04d}-of-{count:04d}.zip"
        with zipfile.ZipFile(output / part, "x", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("chunk.bin", chunk)
        manifest["parts"].append({"name": part, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    with (output / (name + ".parts.json")).open("x") as f:
        json.dump(manifest, f, indent=2)
print("DONE. Open Colab's Files panel and download ALL files inside:", output)
