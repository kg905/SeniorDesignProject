"""Draw measured joints and vectors; never draw a template skeleton."""
import math


def draw_geometry(canvas, output):
    import cv2
    geometry = output.get("geometry") or {}
    points = geometry.get("points", {})
    height, width = canvas.shape[:2]
    face = geometry.get("face_box")
    if face is not None and len(face) == 4 and all(math.isfinite(v) and 0 <= v <= 1 for v in face):
        x1, y1, x2, y2 = [round(v*s) for v,s in zip(face, (width, height, width, height))]
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (20,100,255), 1)
    def pixel(name):
        point = points.get(name, {})
        if not point.get("usable"):
            return None
        x, y = point.get("x"), point.get("y")
        if x is None or y is None or not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return round(x * (width - 1)), round(y * (height - 1))
    segments = [("left_shoulder", "right_shoulder", (0, 230, 255)),
                ("left_hip", "right_hip", (0, 160, 255)),
                ("left_shoulder", "left_hip", (240, 40, 220)),
                ("right_shoulder", "right_hip", (240, 40, 220))]
    for side in ("left", "right"):
        for a, b in (("shoulder", "elbow"), ("elbow", "wrist"), ("hip", "knee"), ("knee", "ankle")):
            segments.append((f"{side}_{a}", f"{side}_{b}", (70, 230, 100)))
    for a, b, color in segments:
        p, q = pixel(a), pixel(b)
        if p is not None and q is not None:
            cv2.arrowedLine(canvas, p, q, color, 2, cv2.LINE_AA, tipLength=.12)
    for name in points:
        p = pixel(name)
        if p is not None:
            cv2.circle(canvas, p, 4, (250, 250, 250), -1, cv2.LINE_AA)
            short = {"left_shoulder":"LS", "right_shoulder":"RS", "left_hip":"LH", "right_hip":"RH"}.get(name)
            if short:
                cv2.putText(canvas, short, (p[0]+5,p[1]+12), cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
    anchors=[pixel(name) for name in ("left_shoulder","right_shoulder","left_hip","right_hip")]
    if all(p is not None for p in anchors):
        ls,rs,lh,rh=anchors
        top=((ls[0]+rs[0])//2,(ls[1]+rs[1])//2)
        bottom=((lh[0]+rh[0])//2,(lh[1]+rh[1])//2)
        cv2.arrowedLine(canvas,top,bottom,(255,190,50),1,cv2.LINE_AA,tipLength=.12)
        cv2.putText(canvas,"T",((top[0]+bottom[0])//2+5,(top[1]+bottom[1])//2),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,190,50),1,cv2.LINE_AA)
    metrics = output.get("metrics", {})
    for name, a, b in (("shoulder_angle_deg", "left_shoulder", "right_shoulder"),
                       ("hip_angle_deg", "left_hip", "right_hip")):
        p, q, value = pixel(a), pixel(b), metrics.get(name)
        if p and q and value is not None and math.isfinite(value):
            prefix="T: " if geometry.get("feature_schema")==3 else ""
            cv2.putText(canvas, f"{prefix}{value:.1f} deg", ((p[0]+q[0])//2, (p[1]+q[1])//2-8),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)


def metric_panel(output, height):
    import cv2
    import numpy as np
    from .features import FEATURES
    panel = np.zeros((max(height, 500), 370, 3), np.uint8)
    basis = "Directed torso angles, not roll" if (output.get("geometry") or {}).get("feature_schema") == 3 else "Angles describe image lines, not roll"
    rows = ["MEASURED GEOMETRY", basis,
            "Source: " + str((output.get("geometry") or {}).get("source", "unavailable"))]
    for name in FEATURES:
        value = output.get("metrics", {}).get(name)
        formatted = f"{value:.3f}" if value is not None and math.isfinite(value) else "N/A"
        rows.append(f"{name}: {formatted}")
    rows += ["POSTURE SCORES"]
    for label in ("back", "side", "prone"):
        value = output.get("scores", {}).get(label)
        rows.append(f"{label}: {value:.1%}" if value is not None else f"{label}: N/A")
    overall = output.get("overall_test_accuracy")
    rows += [f"Overall test score: {100 * overall:.1f}/100" if overall is not None else "Overall test score: N/A",
             f"Monitor fault: {output['monitor_fault']}", f"Baby alarm: {output['baby_alarm']}"]
    for i, row in enumerate(rows):
        cv2.putText(panel, row, (8, 20 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, .39,
                    (220, 230, 230), 1, cv2.LINE_AA)
    return panel
