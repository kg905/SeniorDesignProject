"""Latest-frame Picamera2/OpenCV adapter shared by legacy and single-model capture."""

import threading
import time


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
