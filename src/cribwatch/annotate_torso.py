"""Desktop annotation of a real body box and four verified anatomical anchors."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import read_csv
from .display import read_rgb

ORDER = ("LEFT shoulder", "RIGHT shoulder", "LEFT hip", "RIGHT hip")


def annotate(manifest, output):
    import cv2
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    saved = {}
    if output.exists():
        with output.open() as f:
            for line in f:
                row = json.loads(line)
                saved[row["sha256"]] = row
    rows = [r for r in read_csv(manifest) if r.get("training_eligible") == "1"]
    print("Drag the real body box. Then click anatomical LEFT shoulder, RIGHT shoulder, LEFT hip, RIGHT hip.")
    print("Keys: x = next point not visible/unlabeled; u = undo; s = save after 4; n = skip image; q = quit.")
    for row in rows:
        if row["sha256"] in saved:
            if saved[row["sha256"]]["posture"] != row["posture"]:
                raise ValueError("A reviewed posture label changed; reconcile annotations first")
            continue
        path = Path(row["image_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image changed since manifest creation")
        rgb = read_rgb(path)
        h, w = rgb.shape[:2]
        scale = min(1., 960/w, 720/h)
        image = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (round(w*scale), round(h*scale)))
        dh, dw = image.shape[:2]
        title = "Verified body box: ENTER accept, ESC skip"
        x,y,bw,bh = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)
        if bw < 8 or bh < 8:
            continue
        points = []
        title = "Verified anchors"
        cv2.namedWindow(title)
        def click(event, px, py, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
                if x <= px <= x+bw and y <= py <= y+bh:
                    points.append([px/dw,py/dh,2])
        cv2.setMouseCallback(title, click)
        quit_all = False
        while True:
            canvas = image.copy()
            cv2.rectangle(canvas,(x,y),(x+bw,y+bh),(0,220,255),1)
            for i,(px,py,visible) in enumerate(points):
                if visible:
                    pt=(round(px*dw),round(py*dh))
                    cv2.circle(canvas,pt,4,(20,240,60),-1)
                    cv2.putText(canvas,str(i+1),pt,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            instruction = f"Click {ORDER[len(points)]}" if len(points)<4 else "s = save verified points"
            cv2.putText(canvas,instruction,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            cv2.imshow(title,canvas)
            key=cv2.waitKey(20)&0xff
            if key==ord('q'):
                quit_all=True;break
            if key==ord('n'):
                break
            if key==ord('u') and points:
                points.pop()
            if key==ord('x') and len(points)<4:
                points.append([0.,0.,0])
            if key==ord('s') and len(points)==4:
                record={k:row.get(k,"") for k in ("image_path","sha256","posture","source_group","session_id","baby_id","original_image_group","subject_kind")}
                record.update(box=[(x+bw/2)/dw,(y+bh/2)/dh,bw/dw,bh/dh],keypoints=points,
                              annotation="human_verified_torso4",keypoint_order=list(ORDER))
                with output.open('a') as f:
                    f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                saved[row['sha256']]=record
                print('Saved verified anchors:',len(saved),flush=True)
                break
        cv2.destroyWindow(title)
        if quit_all:
            break
    return len(saved)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest');parser.add_argument('--output',required=True)
    args=parser.parse_args();annotate(args.manifest,args.output)
