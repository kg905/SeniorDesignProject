"""Explicit crib crop shared by capture and inference; normalized to camera frame."""
import math


def parse_roi(value):
    try:
        coords = tuple(float(v) for v in value.split(","))
    except (ValueError, AttributeError) as error:
        raise ValueError("ROI format: x1,y1,x2,y2 in normalized 0–1 camera coordinates") from error
    if len(coords) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
        raise ValueError("ROI needs four finite 0–1 values")
    if coords[0] >= coords[2] or coords[1] >= coords[3]:
        raise ValueError("ROI must have positive width and height")
    return coords


def crop_frame(rgb, roi):
    if roi is None:
        return rgb
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [round(value * scale) for value, scale in zip(roi, (w, h, w, h))]
    if x2-x1 < 16 or y2-y1 < 16:
        raise ValueError("Crib ROI is too small")
    return rgb[y1:y2, x1:x2].copy()
