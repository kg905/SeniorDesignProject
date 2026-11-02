"""Build a grouped four-keypoint YOLO dataset from real human annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .features import LABELS
from .learning import group_split

ORDER=("LEFT shoulder","RIGHT shoulder","LEFT hip","RIGHT hip")


def prepare(annotations,output,*,split_manifest=None):
    import csv
    output=Path(output)
    if output.exists():
        raise FileExistsError("Preserve the existing pose dataset; choose a fresh output path")
    rows=[];seen=set()
    with Path(annotations).open() as f:
        for line in f:
            row=json.loads(line)
            if row.get('annotation')!='human_verified_torso4' or tuple(row.get('keypoint_order',[]))!=ORDER:
                raise ValueError("Need verified left/right shoulder/hip annotations, not model-generated guesses")
            if row['sha256'] in seen or row.get('posture') not in LABELS or not row.get('source_group'):
                raise ValueError("Duplicate/missing posture or source grouping")
            seen.add(row['sha256'])
            if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError("Annotated image changed")
            values=row.get('box',[])
            points=row.get('keypoints',[])
            if len(values)!=4 or any(not 0<=v<=1 for v in values) or values[2]<=0 or values[3]<=0:
                raise ValueError("Invalid normalized body box")
            if len(points)!=4 or any(len(p)!=3 or not 0<=p[0]<=1 or not 0<=p[1]<=1 or p[2] not in (0,1,2) for p in points):
                raise ValueError("Invalid four-keypoint annotation")
            rows.append(row)
    if not rows:
        raise ValueError("No verified anchors")
    if split_manifest:
        with Path(split_manifest).open(newline='') as f:
            assigned={r['sha256']:r['split'] for r in csv.DictReader(f)}
        if any(r['sha256'] not in assigned for r in rows):
            raise ValueError("Annotations must share the posture model's declared split; do not guess unknown folds")
        folds=[assigned[r['sha256']] for r in rows]
    else:
        folds=['']*len(rows)
        for name,indices in zip(('train','validation','test'),group_split(rows,group_field='source_group')):
            for i in indices:folds[i]=name
    if set(folds)!={'train','validation','test'}:
        raise ValueError("Need actual train, validation and test groups")
    owners={}
    for row,fold in zip(rows,folds):
        for field in ('source_group','session_id','baby_id','original_image_group'):
            value=row.get(field)
            if value:
                key=(field,value)
                if key in owners and owners[key]!=fold:
                    raise ValueError('Annotation groups cross folds')
                owners[key]=fold
    output.mkdir(parents=True)
    for row,fold in zip(rows,folds):
        images=output/'images'/fold;labels=output/'labels'/fold
        images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
        source=Path(row['image_path']);name=row['sha256']+source.suffix.lower()
        shutil.copy2(source,images/name)
        values=[0,*row['box'],*[v for point in row['keypoints'] for v in point]]
        (labels/(Path(name).stem+'.txt')).write_text(' '.join(str(v) for v in values)+'\n')
    config=output/'torso4.yaml'
    config.write_text(f"path: {json.dumps(str(output.resolve()))}\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: baby_body\nkpt_shape: [4, 3]\nflip_idx: [1, 0, 3, 2]\n")
    (output/'split.json').write_text(json.dumps([dict(sha256=r['sha256'],source_group=r['source_group'],split=s) for r,s in zip(rows,folds)],indent=2))
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('annotations');p.add_argument('--output',required=True);p.add_argument('--split-manifest')
    p.add_argument('--train',action='store_true');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--models',default='models')
    args=p.parse_args();config=prepare(args.annotations,args.output,split_manifest=args.split_manifest)
    print('Pose dataset:',config)
    if args.train:
        from ultralytics import YOLO
        from .assets import download_models
        # Train the feature extractor only on train groups. YOLO selects its best
        # checkpoint on validation; the YAML test path is not used for selection.
        download_models(args.models,names=['yolov8n-pose.pt'])
        detector=YOLO(str(Path(args.models)/'yolov8n-pose.pt'))
        detector.train(data=str(config),epochs=args.epochs,imgsz=416,batch=16,
            project=str(Path(args.output)/'training'),name='torso4',exist_ok=False,seed=42)
