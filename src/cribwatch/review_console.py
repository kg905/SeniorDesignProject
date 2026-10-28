"""Colab labeling using ordinary image output + input(), without widget rendering."""

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

from .data import read_csv, write_csv

POSTURES = {"b": "back", "back": "back", "s": "side", "side": "side", "p": "prone",
            "prone": "prone", "belly": "prone", "u": "unknown", "unknown": "unknown",
            "unclear": "unknown", "n": "no_baby", "no_baby": "no_baby"}
FACES = {"unknown", "visible", "turned_away", "occluded"}


def save_review(manifest_path, shown_row, posture, *, face=None, baby=None, session=None):
    """Reload before saving, so a stale batch cannot overwrite other saved rows."""
    if posture not in set(POSTURES.values()) or (face is not None and face not in FACES):
        raise ValueError("Unknown review label")
    if hashlib.sha256(Path(shown_row["image_path"]).read_bytes()).hexdigest() != shown_row["sha256"]:
        raise ValueError("Image changed since it was indexed; not saving a label")
    rows = read_csv(manifest_path)
    matches = [r for r in rows if (r["image_path"], r["sha256"]) == (shown_row["image_path"], shown_row["sha256"])]
    if len(matches) != 1:
        raise ValueError("Review row changed/disappeared; rerun the labeling cell")
    row = matches[0]
    if row.get("reviewed") == "1":
        raise ValueError("This image was already saved elsewhere; rerun to load current progress")
    row.update(posture=posture, reviewed="1", human_verified="1", label_source="human_review")
    if face is not None:
        row["face_state"] = face
    if baby is not None:
        row["baby_id"] = baby.strip()
    if session is not None:
        row["session_id"] = session.strip()
    if "training_eligible" in row:
        row["training_eligible"] = str(int(posture != "unknown" and row.get("label_conflict", "0") != "1"))
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(manifest_path, rows, columns)


def review_console(manifest_path, batch_size=12, *, input_fn=None, display_fn=None):
    """Show one image and prompt for a label; save each Enter, q stops without labeling."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if display_fn is None:
        from IPython.display import display
        display_fn = display
    if input_fn is None:
        input_fn = input
    rows = read_csv(manifest_path)
    pending = [r for r in rows if r.get("reviewed") != "1"]
    reviewed = len(rows) - len(pending)
    print(f"Review file: {manifest_path}\nAlready saved: {reviewed} | Remaining: {len(pending)}", flush=True)
    if not pending:
        print("No unreviewed images remain. Continue to the export section.", flush=True)
        return 0
    print("Type b=back, s=side, p=belly/prone, u=unclear, n=no baby, q=stop. Press Enter to save each label.", flush=True)
    saved = 0
    for row in pending[:batch_size]:
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Image is missing: {path}. Rerun notebook sections 1 and 3; keep the Drive CSV.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image changed since indexing; not showing it as the indexed image")
        print(f"\nImage {saved + 1} of this batch | Group: {row.get('source_group', 'unassigned')}", flush=True)
        if row.get("label_conflict") == "1":
            print("Publisher-label conflict: review this original-source group; unresolved groups stay quarantined.", flush=True)
        with Image.open(path) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((760, 520))
            display_fn(preview.copy())
        while True:
            try:
                answer = input_fn("Label [b/s/p/u/n/q], then Enter: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print(f"\nStopped. {saved} labels saved; rerun this cell to continue.", flush=True)
                return saved
            if answer in {"q", "quit", "stop"}:
                print(f"Stopped. {saved} labels saved; this image is still pending.", flush=True)
                return saved
            parts = answer.split()
            if parts and parts[0] in POSTURES and len(parts) <= 2 and (len(parts) == 1 or parts[1] in FACES):
                posture = POSTURES[parts[0]]
                face = parts[1] if len(parts) == 2 else None
                break
            print("Please type b, s, p, u, n, or q. Nothing was saved for that entry.", flush=True)
        save_review(manifest_path, row, posture, face=face)
        saved += 1
        print(f"Saved: {posture}. Total saved: {reviewed + saved}/{len(rows)}.", flush=True)
    print(f"Batch complete: {saved} saved. Rerun this cell for the next batch, or continue to export.", flush=True)
    return saved
