"""MediaPipe + one 12-metric classifier with annotated live/offline scans."""

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np

from .policy import AlarmPolicy, PolicyConfig
from .camera import LatestCamera
from .transport import get_key
from .display import draw_overlay, make_output, read_rgb
from .metrics_model import MetricsPredictor
from .tcp_alarm import TCPAlarmClient


class InferenceWorker:
    """A hung inference cannot freeze the GUI/watchdog or sustain a reassuring result."""
    def __init__(self, camera, predictor):
        self.camera, self.predictor = camera, predictor
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.result = None
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        seen = -1
        while not self.stop.is_set():
            frame = self.camera.get()
            if not frame or frame[0] == seen or time.monotonic() - frame[1] > 1:
                self.stop.wait(.02)
                continue
            seen = frame[0]
            start = time.monotonic()
            try:
                prediction = self.predictor.predict(frame[2])
            except Exception:
                prediction = ("unknown", 0., {}, "inference_error", {}, None)
            with self.lock:
                self.result = (frame[0], frame[1], frame[2], prediction, (time.monotonic() - start) * 1000)

    def get(self):
        with self.lock:
            return self.result

    def close(self):
        self.stop.set()
        self.worker.join(timeout=1)
        if not self.worker.is_alive():
            self.predictor.close()


def live(args):
    import cv2
    predictor = MetricsPredictor(args.model, models_dir=args.models, showcase=args.showcase)
    client = TCPAlarmClient(args.alarm_host, get_key(), port=args.alarm_port) if args.alarm_host else None
    policy = AlarmPolicy(PolicyConfig(alarm_risk=args.alarm_risk))
    lock, stop = threading.Lock(), threading.Event()
    shared = {"output": make_output("unknown", 0., "starting"), "updated": 0.,
              "link": "connecting" if client else "disabled"}

    def send():
        while not stop.is_set():
            with lock:
                output, updated = shared["output"].copy(), shared["updated"]
            if time.monotonic() - updated > 3:
                output.update(posture="unknown", risk="unknown", confidence=0., reason="processing_stalled", monitor_fault=True,
                              bounding_box=None, metrics={})
                output["state"] = "check_baby" if output["baby_alarm"] else "monitor_fault"
            if client:
                try:
                    client.send(output)
                    link = "connected"
                except Exception:
                    link = "FAILED"
                with lock:
                    shared["link"] = link
            stop.wait(.5)

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    camera = inference = writer = None
    record_size = None
    log = None
    last_snapshot = False
    try:
        if args.log:
            Path(args.log).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.log, "x")
        camera = LatestCamera(args.source, picamera=args.picamera)
        inference = InferenceWorker(camera, predictor)
        start_run = time.monotonic()
        seen_result = -1
        status = policy.update("unknown", start_run)
        while args.duration is None or time.monotonic() - start_run < args.duration:
            start = time.monotonic()
            result = inference.get()
            posture, confidence, scores, reason, elapsed = "unknown", 0., {}, "camera_or_inference_stale", None
            metrics, box = {}, None
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            # Use the exact frame classified, never attach an old prediction to a new image.
            if result and start - result[1] <= 1.5:
                _, _, rgb, prediction, elapsed = result
                posture, confidence, scores, reason, metrics, box = prediction
            ack = bool(args.ack_file and Path(args.ack_file).exists())
            new_result = result is not None and result[0] != seen_result
            if new_result or reason == "camera_or_inference_stale":
                if ack:
                    Path(args.ack_file).unlink()
                status = policy.update(posture, start, acknowledge=ack)
                if result:
                    seen_result = result[0]
            if reason == "camera_or_inference_stale":
                status = dict(status, monitor_fault=True,
                              state="check_baby" if status["baby_alarm"] else "monitor_fault")
            output = make_output(posture, confidence, reason, status=status, scores=scores, inference_ms=elapsed)
            output["experimental_showcase"] = bool(args.showcase)
            output["model_sha256"] = predictor.metadata["model_sha256"]
            output.update(metrics=metrics, bounding_box=box, alarm_trigger_risk=args.alarm_risk)
            with lock:
                shared["output"], shared["updated"] = output, start
                link = shared["link"]
            display = draw_overlay(rgb, output, showcase=args.showcase, link=link)
            line = json.dumps(dict(output, alarm_link=link), allow_nan=False)
            print(line, flush=True)
            if log:
                log.write(line + "\n")
                log.flush()
            if args.record:
                if writer is None:
                    path = Path(args.record)
                    if path.exists():
                        raise FileExistsError("Preserving existing recording")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                             (display.shape[1], display.shape[0]))
                    record_size = (display.shape[1], display.shape[0])
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open annotated recording")
                recorded = display
                if (display.shape[1], display.shape[0]) != record_size:
                    # Driver resolutions can differ from requested size; keep valid MP4 frames.
                    scale = min(record_size[0] / display.shape[1], record_size[1] / display.shape[0])
                    small = cv2.resize(display, (max(1, round(display.shape[1] * scale)),
                                                 max(1, round(display.shape[0] * scale))))
                    recorded = np.zeros((record_size[1], record_size[0], 3), np.uint8)
                    y, x = (record_size[1] - small.shape[0]) // 2, (record_size[0] - small.shape[1]) // 2
                    recorded[y:y + small.shape[0], x:x + small.shape[1]] = small
                writer.write(recorded)
            key = -1
            if args.display:
                cv2.imshow("CribWatch - experimental posture demonstration", display)
                key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") or (args.snapshot and not last_snapshot and posture != "unknown"):
                path = Path(args.snapshot or f"snapshots/scan-{time.time_ns()}.jpg")
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise FileExistsError("Preserving previous snapshot")
                if not cv2.imwrite(str(path), display):
                    raise RuntimeError("Cannot save snapshot")
                path.with_suffix(".json").write_text(json.dumps(dict(output, alarm_link=link), indent=2))
                last_snapshot = True
            stop.wait(max(0, 1 / args.fps - (time.monotonic() - start)))
    finally:
        # Final best-effort explicit fault; receiver timeout is the fallback on process loss.
        stop.set()
        sender.join(timeout=2)
        if client:
            try:
                with lock:
                    final = shared["output"].copy()
                final.update(posture="unknown", risk="unknown", confidence=0., monitor_fault=True, reason="monitor_stopped")
                final["state"] = "check_baby" if final["baby_alarm"] else "monitor_fault"
                client.send(final)
            except Exception:
                pass
        if inference:
            inference.close()
        else:
            predictor.close()
        if camera:
            camera.close()
        if writer:
            writer.release()
        if log:
            log.close()
        if args.display:
            cv2.destroyAllWindows()


def offline(args):
    import cv2
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    predictor = MetricsPredictor(args.model, models_dir=args.models, showcase=args.showcase)
    try:
        for i, path in enumerate(args.images):
            rgb = read_rgb(path)
            start = time.monotonic()
            posture, confidence, scores, reason, metrics, box = predictor.predict(rgb)
            output = make_output(posture, confidence, reason, scores=scores, inference_ms=(time.monotonic() - start) * 1000)
            output.update(image=Path(path).name, model_sha256=predictor.metadata["model_sha256"], experimental_showcase=args.showcase,
                          metrics=metrics, bounding_box=box)
            result_path = output_dir / f"{i:03d}-{Path(path).stem}.jpg"
            if not cv2.imwrite(str(result_path), draw_overlay(rgb, output, showcase=args.showcase)):
                raise RuntimeError("Cannot save annotated image")
            results.append(output)
    finally:
        predictor.close()
    (output_dir / "predictions.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="One-model posture showcase; not medical risk estimation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("live", "image"):
        p = commands.add_parser(name)
        p.add_argument("--model", required=True, help="One posture .joblib file, or handoff models folder")
        p.add_argument("--models", default="models", help="Official MediaPipe pose/face asset folder")
        p.add_argument("--showcase", action="store_true", help="Permit an evaluated exploratory model, explicitly labeled experimental")
        if name == "live":
            p.add_argument("--picamera", action="store_true")
            p.add_argument("--source", default="0")
            p.add_argument("--display", action="store_true", help="Open an HDMI/desktop OpenCV window")
            p.add_argument("--fps", type=float, default=5)
            p.add_argument("--duration", type=float, help="Stop after this many wall-clock seconds")
            p.add_argument("--alarm-host")
            p.add_argument("--alarm-port", type=int, default=5000)
            p.add_argument("--alarm-risk", choices=["moderate", "high"], default="moderate", help="Minimum posture-risk category that triggers a timed alert")
            p.add_argument("--ack-file")
            p.add_argument("--record", help="Opt-in annotated MP4; wall-clock timing is in JSONL")
            p.add_argument("--log", help="New JSONL file with real timestamps")
            p.add_argument("--snapshot", help="Save the first accepted frame + JSON; s also saves a snapshot")
        else:
            p.add_argument("images", nargs="+")
            p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "live":
        if not np.isfinite(args.fps) or args.fps <= 0 or (args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0)):
            parser.error("fps/duration must be finite positive values")
        live(args)
    else:
        offline(args)


if __name__ == "__main__":
    main()
