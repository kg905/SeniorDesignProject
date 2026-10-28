"""MediaPipe Tasks adapter. Models are explicit local files, never silently fetched."""

from pathlib import Path

from .features import extract_metrics, unknown


class Vision:
    def __init__(self, model_dir, *, pose_name="pose_landmarker_lite.task"):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mp = mp
        self.pose = self.face = None
        model_dir = Path(model_dir)
        for name in (pose_name, "face_landmarker.task"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Missing {model_dir / name}; run cribwatch download-models")
        try:
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / pose_name)),
                running_mode=vision.RunningMode.IMAGE, num_poses=2,
                min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
            ))
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
                running_mode=vision.RunningMode.IMAGE, num_faces=2,
            ))
        except Exception:
            self.close()
            raise

    def observe(self, rgb):
        return self._observe_prepared(self._prepare(rgb))

    @staticmethod
    def _prepare(rgb):
        import cv2
        import numpy as np
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB image H x W x 3")
        # The supplied posture export was stretched to 640 x 640. Apply the
        # same transform at runtime; otherwise geometric ratios shift with aspect.
        if rgb.shape[:2] != (640, 640):
            rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        return rgb

    def observe_region(self, rgb, region):
        """Infer on a crop, remapping 2D landmarks to the original 640x640 frame."""
        import cv2
        import numpy as np
        rgb = self._prepare(rgb)
        bounds = np.array(region, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            return unknown("invalid_body_region")
        x1, y1 = np.maximum(0, np.floor(bounds[:2] * 640)).astype(int)
        x2, y2 = np.minimum(640, np.ceil(bounds[2:] * 640)).astype(int)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return unknown("body_region_too_small")
        crop = cv2.resize(rgb[y1:y2, x1:x2], (640, 640), interpolation=cv2.INTER_LINEAR)
        return self._observe_prepared(crop, region=(x1 / 640, y1 / 640, x2 / 640, y2 / 640))

    def face_summary(self, rgb):
        """Dense face landmarks can be measured even when a body pose is missing."""
        import numpy as np
        rgb = self._prepare(rgb)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        face = self.face.detect(image)
        count = len(face.face_landmarks)
        return count, len(face.face_landmarks[0]) if count == 1 else 0

    def _observe_prepared(self, rgb, region=None):
        import numpy as np
        import copy
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pose = self.pose.detect(image)
        if len(pose.pose_landmarks) != 1:
            return unknown("no_pose" if not pose.pose_landmarks else "multiple_people")
        face = self.face.detect(image)
        if len(face.face_landmarks) > 1:
            return unknown("multiple_faces")
        landmarks = pose.pose_landmarks[0]
        if region is not None:
            x1, y1, x2, y2 = region
            landmarks = [copy.copy(point) for point in landmarks]
            for point in landmarks:
                point.x, point.y = x1 + point.x * (x2 - x1), y1 + point.y * (y2 - y1)
        return extract_metrics(
            landmarks, 640, 640,
            world=pose.pose_world_landmarks[0] if pose.pose_world_landmarks else None,
            face_count=len(face.face_landmarks),
            face_landmark_count=len(face.face_landmarks[0]) if face.face_landmarks else 0,
        )

    def close(self):
        for task in (self.pose, self.face):
            if task is not None:
                task.close()
        self.pose = self.face = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_vision(model_dir, *, pose_name="pose_landmarker_lite.task", adapter=None):
    if adapter is None:
        return Vision(model_dir, pose_name=pose_name)
    from .hybrid_vision import HybridVision
    return HybridVision(model_dir, pose_name=pose_name, adapter=adapter)
