"""Live vision loop and portable alarm. Logs metadata only; no raw video saving."""

import json
import math
from pathlib import Path
import threading
import time

from .features import unknown
from .policy import AlarmPolicy
from .transport import AlarmClient, get_key
from .camera import LatestCamera
from .alarm_runtime import receiver
from .vision import Vision


def monitor(args):
    from .learning import Predictor
    from .assets import verify_classifier_assets
    predictor = Predictor(args.classifier)
    pose_name = verify_classifier_assets(predictor.artifact, args.models)
    policy = AlarmPolicy()
    client = AlarmClient(args.alarm_url, get_key()) if args.alarm_url else None
    if args.alarm_url is None:
        print("LOCAL RESEARCH MODE: no physical alarm is connected", flush=True)
    shared = {"status": {"state": "monitor_fault", "baby_alarm": False, "monitor_fault": True}, "updated": 0.0}
    lock, stop = threading.Lock(), threading.Event()

    def deliver():
        while not stop.is_set():
            with lock:
                status = shared["status"].copy()
                stale = time.monotonic() - shared["updated"] > 3.0
            if stale:
                status["monitor_fault"] = True
                status["state"] = "monitor_fault" if not status["baby_alarm"] else "check_baby"
            if client:
                try:
                    client.send(status)
                except Exception as e:
                    # Never log a URL, shared key, request body, or infant image.
                    print(json.dumps({"event": "alarm_link_error", "error_type": type(e).__name__}), flush=True)
            stop.wait(.5)

    sender = threading.Thread(target=deliver, daemon=True)
    sender.start()
    camera = None
    try:
        with Vision(args.models, pose_name=pose_name) as vision:
            camera = LatestCamera(args.source, picamera=args.picamera)
            seen = -1
            while True:
                start = time.monotonic()
                frame = camera.get()
                scores = {}
                observation = unknown("camera_stale_or_missing")
                posture, reason = "unknown", observation.reason
                if frame and frame[0] != seen and start - frame[1] <= 1.0:
                    seen = frame[0]
                    observation = vision.observe(frame[2])
                    posture, scores, reason = predictor.predict(observation, rgb=frame[2])
                    if time.monotonic() - frame[1] > policy.config.max_gap_seconds:
                        posture, reason = "unknown", "inference_too_slow"
                ack_path = Path(args.ack_file) if args.ack_file else None
                acknowledge = bool(ack_path and ack_path.exists())
                if acknowledge:
                    ack_path.unlink()
                status = policy.update(posture, time.monotonic(), acknowledge=acknowledge)
                with lock:
                    shared["status"], shared["updated"] = status, time.monotonic()
                metrics = {k: v if isinstance(v, (int, float)) and math.isfinite(v) else None
                           for k, v in observation.metrics.items()}
                print(json.dumps({**status, "reason": reason, "model_scores": scores, "metrics": metrics,
                                  "face_state": observation.face_state,
                                  "processing_ms": round((time.monotonic() - start) * 1000, 1)}, allow_nan=False), flush=True)
                stop.wait(max(0, 1 / args.fps - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        sender.join(timeout=2)
        if camera is not None:
            camera.close()
        # Receiver independently alarms after heartbeat timeout when this process stops.
