"""Live vision loop and portable alarm. Logs metadata only; no raw video saving."""

import json
import math
import os
from pathlib import Path
import threading
import time

from .features import unknown
from .learning import Predictor
from .policy import AlarmPolicy
from .transport import AlarmClient, ReceiverState, make_server
from .vision import Vision


class LatestCamera:
    """Read on a daemon thread so stalls don't block the observation watchdog."""
    def __init__(self, source, *, picamera=False):
        import cv2
        self.cv2 = cv2
        self.picamera = picamera
        self.camera = None
        if picamera:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
            self.camera.start()
        else:
            self.camera = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            if not self.camera.isOpened():
                self.camera.release()
                raise RuntimeError("Cannot open camera/video source")
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.stopped = threading.Event()
        self.worker = threading.Thread(target=self._read, daemon=True)
        self.worker.start()

    def _read(self):
        while not self.stopped.is_set():
            try:
                if self.picamera:
                    # libcamera RGB888 exposes BGR byte order on little-endian Pi.
                    bgr = self.camera.capture_array("main")
                    ok = True
                else:
                    ok, bgr = self.camera.read()
                if not ok:
                    self.stopped.wait(.05)
                    continue
                rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.sequence += 1
                    self.latest = (self.sequence, time.monotonic(), rgb)
            except Exception:
                self.stopped.wait(.1)

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=1)
        # Releasing a device while its driver is blocked reading can deadlock.
        if not self.worker.is_alive():
            if self.picamera:
                self.camera.stop()
                self.camera.close()
            else:
                self.camera.release()


def get_key():
    key = os.environ.get("ALARM_SHARED_KEY", "").encode()
    from .transport import validate_key
    return validate_key(key)


def monitor(args):
    predictor = Predictor(args.classifier)
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
        with Vision(args.models) as vision:
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
                    posture, scores, reason = predictor.predict(observation)
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


def receiver(args):
    state = ReceiverState(get_key(), timeout=args.timeout)
    server = make_server(args.host, args.port, state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    buzzer = button = None
    muted_until = [0.0]
    if args.gpio is not None:
        from gpiozero import DigitalOutputDevice
        buzzer = DigitalOutputDevice(args.gpio, initial_value=False)
    if args.mute_button is not None:
        from gpiozero import Button
        button = Button(args.mute_button, pull_up=True, bounce_time=.1)
        # Temporary local silence never clears the camera's latched alarm.
        button.when_pressed = lambda: muted_until.__setitem__(0, time.monotonic() + 10)
    worker.start()
    previous = None
    try:
        while True:
            status = state.snapshot()
            audible = status["audible"] and time.monotonic() >= muted_until[0]
            # Continuous tone for baby alert; pulsed tone for monitor/link fault.
            on = audible and (status["baby_alarm"] or time.monotonic() % 1 < .35)
            if buzzer:
                buzzer.value = on
            if status != previous:
                print(json.dumps(status), flush=True)
                previous = status
            time.sleep(.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if buzzer:
            buzzer.off()
            buzzer.close()
        if button:
            button.close()
