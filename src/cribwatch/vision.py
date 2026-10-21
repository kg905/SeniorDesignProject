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
        import cv2
        import numpy as np
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB image H x W x 3")
        # The supplied posture export was stretched to 640 x 640. Apply the
        # same transform at runtime; otherwise geometric ratios shift with aspect.
        if rgb.shape[:2] != (640, 640):
            rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        pose = self.pose.detect(image)
        if len(pose.pose_landmarks) != 1:
            return unknown("no_pose" if not pose.pose_landmarks else "multiple_people")
        face = self.face.detect(image)
        if len(face.face_landmarks) > 1:
            return unknown("multiple_faces")
        return extract_metrics(
            pose.pose_landmarks[0], rgb.shape[1], rgb.shape[0],
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
