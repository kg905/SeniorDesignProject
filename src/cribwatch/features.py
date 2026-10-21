"""Geometry in image pixels; distances normalized to projected torso length."""

from dataclasses import dataclass
import math

import numpy as np

FEATURES = (
    "shoulder_vec_mag", "hip_vec_mag", "shoulder_angle_deg", "hip_angle_deg",
    "shoulder_hip_angle_diff", "left_shoulder_visibility", "right_shoulder_visibility",
    "left_hip_visibility", "right_hip_visibility", "face_detected",
    "face_landmark_count", "head_lift_ratio",
)
LABELS = ("back", "side", "prone")
SCHEMA_VERSION = 2
VISION_PREPROCESSING = "stretch_640x640_rgb_v1"


@dataclass(frozen=True)
class Observation:
    metrics: dict
    pose_ok: bool
    reason: str
    face_state: str = "unknown"


def unknown(reason):
    return Observation(dict.fromkeys(FEATURES, math.nan), False, reason)


def axial_angle(vec):
    """Undirected line orientation, degrees in [-90, 90)."""
    return float((np.degrees(np.arctan2(vec[1], vec[0])) + 90) % 180 - 90)


def extract_metrics(landmarks, width, height, *, world=None, face_count=0,
                    face_landmark_count=0, min_visibility=0.5):
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if landmarks is None or len(landmarks) < 33:
        return unknown("no_pose")
    # MediaPipe Pose indices: nose=0; shoulders=11/12; hips=23/24.
    ids = (11, 12, 23, 24)
    xy = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in ids])
    visibility = np.array([landmarks[i].visibility for i in ids], dtype=float)
    presence = np.array([getattr(landmarks[i], "presence", 1.0) for i in ids], dtype=float)
    if not np.isfinite(xy).all() or not np.isfinite(visibility).all() or not np.isfinite(presence).all():
        return unknown("invalid_landmarks")
    ls, rs, lh, rh = xy
    torso = np.linalg.norm((ls + rs - lh - rh) / 2)
    if torso < max(width, height) * 0.02:
        return unknown("torso_too_small")
    shoulder, hip = rs - ls, rh - lh
    sa, ha = axial_angle(shoulder), axial_angle(hip)
    metrics = dict(zip(FEATURES[:9], [
        float(np.linalg.norm(shoulder) / torso), float(np.linalg.norm(hip) / torso),
        sa, ha, abs((sa - ha + 90) % 180 - 90), *visibility.tolist(),
    ]))
    metrics["face_detected"] = float(face_count == 1)
    metrics["face_landmark_count"] = float(face_landmark_count if face_count == 1 else 0)
    metrics["head_lift_ratio"] = math.nan
    if world is not None and len(world) >= 33 and face_count == 1 and landmarks[0].visibility >= min_visibility:
        xyz = np.array([[p.x, p.y, p.z] for p in world])
        mid_shoulder, mid_hip = xyz[[11, 12]].mean(0), xyz[[23, 24]].mean(0)
        length = np.linalg.norm(mid_shoulder - mid_hip)
        if np.isfinite(xyz[[0, 11, 12, 23, 24]]).all() and length > 1e-6:
            # Camera-depth proxy ONLY. Not a measurement of elevation or airway clearance.
            metrics["head_lift_ratio"] = float((mid_shoulder[2] - xyz[0, 2]) / length)
    in_frame = bool(((xy >= 0) & (xy <= [width, height])).all())
    good = bool(in_frame and (visibility >= min_visibility).all() and (presence >= min_visibility).all())
    return Observation(metrics, good, "ok" if good else "low_pose_quality",
                       "visible" if face_count == 1 else "unknown")


def feature_vector(metrics):
    result = np.array([float(metrics.get(name, math.nan)) for name in FEATURES])
    result[~np.isfinite(result)] = np.nan
    return result
