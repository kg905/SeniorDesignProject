import hashlib

from PIL import Image
import pytest

from cribwatch.data import read_csv, write_csv
from cribwatch.review_console import review_console, save_review


def manifest(tmp_path, count=3):
    rows = []
    for i in range(count):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (30, 20), (i * 30, 50, 100)).save(path)
        rows.append(dict(image_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         posture="unknown", reviewed="0", source_group=f"g-{i}", face_state="unknown",
                         baby_id="", session_id="", label_conflict="0", training_eligible="0"))
    csv = tmp_path / "review.csv"
    write_csv(csv, rows, list(rows[0]))
    return csv


def test_console_saves_each_answer_and_resumes_without_widget_dependencies(tmp_path):
    csv = manifest(tmp_path)
    shown = []
    answers = iter(["typo", "b", "q"])
    assert review_console(csv, input_fn=lambda _: next(answers), display_fn=shown.append) == 1
    rows = read_csv(csv)
    assert rows[0]["posture"] == "back" and rows[0]["reviewed"] == "1"
    assert rows[1]["reviewed"] == "0" and len(shown) == 2
    first = rows[0].copy()
    answers = iter(["p visible", "u"])
    assert review_console(csv, input_fn=lambda _: next(answers), display_fn=shown.append) == 2
    rows = read_csv(csv)
    assert rows[0] == first
    assert rows[1]["posture"] == "prone" and rows[1]["face_state"] == "visible"
    assert rows[2]["reviewed"] == "1" and rows[2]["training_eligible"] == "0"
    assert all(r["label_source"] == "human_review" for r in rows)
    assert review_console(csv, input_fn=lambda _: pytest.fail("Must not prompt"), display_fn=shown.append) == 0


def test_quit_or_interrupt_does_not_label_current_image(tmp_path):
    csv = manifest(tmp_path)
    def interrupt(_):
        raise KeyboardInterrupt
    assert review_console(csv, input_fn=interrupt, display_fn=lambda _: None) == 0
    assert all(r["reviewed"] == "0" for r in read_csv(csv))


def test_changed_image_is_not_shown_or_labeled(tmp_path):
    csv = manifest(tmp_path)
    row = read_csv(csv)[0]
    Image.new("RGB", (10, 10)).save(row["image_path"])
    with pytest.raises(ValueError, match="changed"):
        review_console(csv, input_fn=lambda _: "b", display_fn=lambda _: pytest.fail("Do not show mismatched image"))
    assert read_csv(csv)[0]["reviewed"] == "0"


def test_stale_save_preserves_labels_saved_elsewhere(tmp_path):
    csv = manifest(tmp_path)
    stale = read_csv(csv)
    save_review(csv, stale[2], "side")
    save_review(csv, stale[0], "back")
    rows = read_csv(csv)
    assert rows[2]["posture"] == "side" and rows[0]["posture"] == "back"
    with pytest.raises(ValueError, match="already saved"):
        save_review(csv, stale[0], "prone")


def test_widget_tree_contains_images_without_output_capture_and_saves(tmp_path, monkeypatch):
    import IPython.display
    import ipywidgets as w
    from cribwatch.data import review_widget
    csv = manifest(tmp_path, count=2)
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    container = review_widget(csv, batch_size=1)
    assert displayed == [container] and isinstance(container, w.VBox)
    image_controls = container.children[1].children
    assert isinstance(image_controls[1], w.Image) and image_controls[1].value
    assert not any(isinstance(child, w.Output) for child in container.children)
    image_controls[2].value = "side"
    container.children[-2].click()
    assert read_csv(csv)[0]["posture"] == "side"
    assert "Saved: 1 / 2" in container.children[0].value
