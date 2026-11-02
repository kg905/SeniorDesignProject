import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from cribwatch.data import write_csv
from cribwatch.torso_pose import ORDER, prepare


def annotations(tmp_path):
    # Artificial coordinates verify format/grouping only, not actual keypoint accuracy.
    rows=[]
    for group in range(20):
        for index,label in enumerate(('back','side','prone')):
            path=tmp_path/f'{group}-{label}.png'
            Image.new('RGB',(20,20),(group+10,index+30,100)).save(path)
            rows.append(dict(image_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                posture=label,source_group=f'g-{group}',session_id=f's-{group}',baby_id='',original_image_group='',
                box=[.5,.5,.8,.8],keypoints=[[.3,.3,2],[.7,.3,2],[.35,.7,2],[.65,.7,2]],
                keypoint_order=list(ORDER),annotation='human_verified_torso4'))
    path=tmp_path/'anchors.jsonl'
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    return path,rows


def test_verified_pose_export_has_four_points_and_disjoint_groups(tmp_path):
    path,rows=annotations(tmp_path)
    config=prepare(path,tmp_path/'export')
    assert 'kpt_shape: [4, 3]' in config.read_text()
    label=next((config.parent/'labels/train').glob('*.txt')).read_text().split()
    assert len(label)==17 and label[0]=='0'
    split=json.loads((config.parent/'split.json').read_text())
    owners={}
    for row in split:
        if row['source_group'] in owners:assert owners[row['source_group']]==row['split']
        owners[row['source_group']]=row['split']


def test_given_posture_split_cannot_leak_sessions_between_folds(tmp_path):
    path,rows=annotations(tmp_path)
    for row in rows:row['session_id']='same-real-session'
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    split=[dict(sha256=r['sha256'],split=('train','validation','test')[i%3]) for i,r in enumerate(rows)]
    manifest=tmp_path/'split.csv';write_csv(manifest,split,list(split[0]))
    with pytest.raises(ValueError,match='cross folds'):
        prepare(path,tmp_path/'invalid',split_manifest=manifest)
    assert not (tmp_path/'invalid').exists()
