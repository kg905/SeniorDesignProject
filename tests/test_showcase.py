import hashlib
import json
import threading
from types import SimpleNamespace as P

import numpy as np
from PIL import Image
import pytest

from cribwatch.data import read_csv, write_csv
from cribwatch.display import draw_overlay, make_output
from cribwatch.features import extract_metrics
from cribwatch.labeling import export_reviewed, prepare_review, rebase_reviewed, resolve_reviewed_conflicts
from cribwatch.policy import AlarmPolicy, PolicyConfig
from cribwatch.tcp_alarm import TCPAlarmClient, canonical, make_tcp_server
from cribwatch.transport import ReceiverState


def test_pose_box_uses_visible_points_and_bad_pose_has_no_box():
    points = [P(x=.5, y=.2, z=0., visibility=.9, presence=.9) for _ in range(33)]
    for i, x, y in [(11, .3, .3), (12, .7, .3), (23, .35, .7), (24, .65, .7)]:
        points[i] = P(x=x, y=y, z=0., visibility=.9, presence=.9)
    observation = extract_metrics(points, 640, 640)
    assert observation.pose_ok
    x1, y1, x2, y2 = observation.pose_box
    assert 0 <= x1 < .3 < .7 < x2 <= 1
    assert 0 <= y1 < .2 < .7 < y2 <= 1
    points[11].visibility = .1
    assert extract_metrics(points, 640, 640).pose_box is None


def test_overlay_draws_pose_region_and_preserves_source_frame():
    rgb = np.zeros((100, 200, 3), np.uint8)
    original = rgb.copy()
    output = dict(make_output("prone", .9, "ok"), bounding_box=[.1, .2, .9, .8])
    annotated = draw_overlay(rgb, output)
    assert annotated.shape == (245, 720, 3)
    assert annotated[165, 20, 2] > 0  # Actual rectangle in the photo, below the banner.
    assert np.array_equal(rgb, original)
    assert make_output("unknown", 0., "no_pose")["risk"] == "unknown"


def test_configured_risk_threshold_drives_the_latched_alarm():
    high = AlarmPolicy(PolicyConfig(alarm_risk="high"))
    for t in (0., 1., 2., 3.):
        assert high.update("side", t)["baby_alarm"] is False
    assert high.update("prone", 3.1)["baby_alarm"] is False
    assert high.update("prone", 4.2)["baby_alarm"] is True
    assert high.update("back", 4.3)["baby_alarm"] is True
    moderate = AlarmPolicy(PolicyConfig(alarm_risk="moderate"))
    for t in (0., 1., 2.1):
        status = moderate.update("side", t)
    assert status["baby_alarm"] is True


def test_below_threshold_breaks_continuity():
    policy = AlarmPolicy(PolicyConfig(alarm_risk="high"))
    policy.update("prone", 0.)
    policy.update("side", .8)
    assert not policy.update("prone", .9)["baby_alarm"]
    assert not policy.update("prone", 1.2)["baby_alarm"]


def test_review_queue_and_export_never_promote_unreviewed_labels(tmp_path):
    rows = []
    for i, posture in enumerate(["back", "unknown", "prone"]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (10, 10), (i * 40, 10, 20)).save(path)
        rows.append(dict(image_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            posture=posture, reviewed="0", label_source="dataset_annotation", source_group=str(i),
            label_conflict="1" if i == 2 else "0"))
    source, review = tmp_path / "source.csv", tmp_path / "review.csv"
    write_csv(source, rows, list(rows[0]))
    assert prepare_review(source, review) == {"0": 1, "1": 1, "2": 1}
    with pytest.raises(ValueError, match="No reviewed"):
        export_reviewed(review, tmp_path / "empty")
    reviewed = read_csv(review)
    reviewed[-1].update(reviewed="1", label_source="human_review")
    write_csv(review, reviewed, list(reviewed[0]))
    assert export_reviewed(review, tmp_path / "export", folders=True) == {"back": 1}
    output = read_csv(tmp_path / "export/reviewed.csv")
    assert len(output) == 1 and (tmp_path / "export/back").is_dir()
    assert read_csv(source)[0]["reviewed"] == "0"
    output[0]["image_path"] = "/content/drive/no-longer-mounted/example.jpg"
    write_csv(tmp_path / "shared.csv", output, list(output[0]))
    assert rebase_reviewed(tmp_path / "shared.csv", source, tmp_path / "rebased.csv") == 1
    assert read_csv(tmp_path / "rebased.csv")[0]["image_path"] == rows[0]["image_path"]


def test_real_tcp_roundtrip_wrong_key_replay_version_and_timeout():
    import hmac
    now = [0.]
    state = ReceiverState(b"x" * 32, clock=lambda: now[0])
    server = make_tcp_server("127.0.0.1", 0, state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        client = TCPAlarmClient("127.0.0.1", b"x" * 32, port=server.server_address[1])
        assert state.snapshot()["monitor_fault"] is True
        status = make_output("back", .9, "ok", status={"state": "back_observed", "baby_alarm": False, "monitor_fault": False})
        client.send(status)
        assert state.snapshot()["monitor_fault"] is False
        with pytest.raises(ConnectionError):
            TCPAlarmClient("127.0.0.1", b"y" * 32, port=server.server_address[1]).send(status)
        with pytest.raises(ConnectionError):
            client.send(dict(status, schema_version=99))
        nonce = client.exchange({"op": "challenge"})["nonce"]
        payload = dict(status, nonce=nonce)
        request = {"op": "status", "payload": payload, "signature": hmac.new(b"x" * 32, canonical(payload), hashlib.sha256).hexdigest()}
        assert client.exchange(request)["accepted"] is True
        assert client.exchange(request)["accepted"] is False
        now[0] = 6.
        assert state.snapshot()["monitor_fault"] is True
    finally:
        server.shutdown()
        server.server_close()


def test_conflicts_require_complete_consistent_group_review(tmp_path):
    rows = [dict(original_image_group="same-source", posture="back", reviewed="1", label_source="human_review", label_conflict="1", training_eligible="0"),
            dict(original_image_group="same-source", posture="prone", reviewed="0", label_source="dataset_annotation", label_conflict="1", training_eligible="0")]
    path = tmp_path / "review.csv"
    write_csv(path, rows, list(rows[0]))
    assert resolve_reviewed_conflicts(path)["resolved_original_groups"] == 0
    rows[1].update(reviewed="1", label_source="human_review")
    write_csv(path, rows, list(rows[0]))
    assert resolve_reviewed_conflicts(path)["resolved_original_groups"] == 0
    rows[1]["posture"] = "back"
    write_csv(path, rows, list(rows[0]))
    assert resolve_reviewed_conflicts(path)["resolved_original_groups"] == 1
    assert all(r["label_conflict"] == "0" for r in read_csv(path))


def test_repeated_frame_never_accumulates_new_alarm_evidence_and_then_faults(tmp_path, monkeypatch):
    import time
    from cribwatch import showcase
    class Camera:
        def __init__(self, *args, **kwargs):
            self.frame = (1, time.monotonic(), np.zeros((50, 50, 3), np.uint8))
        def get(self):
            return self.frame
        def close(self):
            pass
    class Model:
        metadata = {"model_sha256": "test-only"}
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, rgb):
            return "prone", .99, {"prone": .99}, "ok", {}, [.1, .1, .9, .9]
        def close(self):
            pass
    monkeypatch.setattr(showcase, "LatestCamera", Camera)
    monkeypatch.setattr(showcase, "MetricsPredictor", Model)
    log = tmp_path / "log.jsonl"
    args = P(model="unused", models="unused", showcase=True, alarm_host=None, alarm_port=5000,
             alarm_risk="moderate", source="0", picamera=False, log=str(log), duration=1.9,
             ack_file=None, record=str(tmp_path / "scan.mp4"), display=False, snapshot=None, fps=10.)
    showcase.live(args)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(r["posture"] == "prone" for r in rows)
    assert not any(r["baby_alarm"] for r in rows)
    assert rows[-1]["risk"] == "unknown" and rows[-1]["monitor_fault"]
    assert rows[-1]["bounding_box"] is None
    import cv2
    capture = cv2.VideoCapture(args.record)
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows)
    capture.release()


def test_handoff_contains_one_classifier_assets_and_launchable_runtime(tmp_path):
    import importlib.util
    import subprocess
    import sys
    import zipfile
    from pathlib import Path
    from cribwatch.features import FEATURES, LABELS
    from cribwatch.learning import train
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_handoff", root / "deploy/single-model/build_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Artificial feature rows test packaging only, never infant model accuracy.
    rows = []
    for baby in range(20):
        for i, label in enumerate(LABELS):
            row = dict(image_path="synthetic-not-an-image", sha256=f"synthetic-{baby}-{i}",
                       baby_id=f"b-{baby}", session_id=f"s-{baby}", posture=label, reviewed="1", pose_ok="1")
            row.update({name: float(i * 10 + baby * .001) for name in FEATURES})
            rows.append(row)
    csv = tmp_path / "synthetic.csv"
    write_csv(csv, rows, list(rows[0]))
    train(csv, tmp_path / "run")
    archive = tmp_path / "handoff.zip"
    module.build(tmp_path / "run", archive, root=root)
    unpack = tmp_path / "unpack"
    with zipfile.ZipFile(archive) as z:
        assert sum(n.endswith(".joblib") for n in z.namelist()) == 1
        assert "models/pose_landmarker_lite.task" in z.namelist()
        assert not any("image_model.py" in n or n.endswith(".tflite") for n in z.namelist())
        z.extractall(unpack)
    for name, digest in json.loads((unpack / "SHA256SUMS.json").read_text()).items():
        assert hashlib.sha256((unpack / name).read_bytes()).hexdigest() == digest
    subprocess.run([sys.executable, str(unpack / "monitor_pi5.py"), "--help"], check=True, capture_output=True)
    subprocess.run([sys.executable, "-S", str(unpack / "alarm_receiver.py"), "--help"], check=True, capture_output=True)
