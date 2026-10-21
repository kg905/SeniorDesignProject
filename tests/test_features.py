from types import SimpleNamespace as P
import math

import numpy as np
import pytest

from cribwatch.features import FEATURES, axial_angle, extract_metrics


def landmarks():
    points = [P(x=.5, y=.2, z=0, visibility=.9, presence=.9) for _ in range(33)]
    for i, x, y in [(11, .3, .3), (12, .7, .3), (23, .35, .7), (24, .65, .7)]:
        points[i] = P(x=x, y=y, z=0, visibility=.9, presence=.9)
    return points


def test_pixel_aspect_and_scale_invariance():
    p = landmarks()
    a = extract_metrics(p, 640, 480, face_count=1, face_landmark_count=478)
    b = extract_metrics(p, 1280, 960, face_count=1, face_landmark_count=478)
    assert a.pose_ok
    assert a.metrics.keys() == set(FEATURES)
    assert a.metrics["shoulder_vec_mag"] == pytest.approx(256 / 192)
    for key in FEATURES[:-1]:
        assert a.metrics[key] == pytest.approx(b.metrics[key])
    assert math.isnan(a.metrics["head_lift_ratio"])


def test_axial_angle_wrap_and_difference():
    assert axial_angle([1, 0]) == axial_angle([-1, 0]) == 0
    p = landmarks()
    p[11].y, p[12].y, p[23].y, p[24].y = .30, .31, .71, .70
    a = extract_metrics(p, 640, 480)
    assert 0 < a.metrics["shoulder_hip_angle_diff"] < 5


@pytest.mark.parametrize("field,value", [("visibility", .1), ("presence", .1), ("x", -1), ("x", float("nan"))])
def test_bad_torso_cannot_pass_quality_gate(field, value):
    p = landmarks()
    setattr(p[11], field, value)
    assert not extract_metrics(p, 640, 480).pose_ok


def test_no_pose_and_degenerate_torso():
    assert extract_metrics(None, 640, 480).reason == "no_pose"
    p = landmarks()
    p[23], p[24] = p[11], p[12]
    assert extract_metrics(p, 640, 480).reason == "torso_too_small"


def test_head_depth_requires_visible_nose_and_face():
    p, world = landmarks(), landmarks()
    world[0].z = -.2
    assert math.isnan(extract_metrics(p, 640, 480, world=world).metrics["head_lift_ratio"])
    assert extract_metrics(p, 640, 480, world=world, face_count=1).metrics["head_lift_ratio"] == pytest.approx(.5)
    p[0].visibility = .1
    assert math.isnan(extract_metrics(p, 640, 480, world=world, face_count=1).metrics["head_lift_ratio"])
