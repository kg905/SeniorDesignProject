import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from cribwatch.features import FEATURES, LABELS
from cribwatch.fusion_model import MetricFusionNet, body_evidence, encode_metrics, fit_normalizer
from cribwatch.fusion_train import predictions_for, train_fusion
from cribwatch.data import read_csv, write_csv


def test_metric_normalization_and_missingness_are_explicit_and_train_only():
    train = [dict.fromkeys(FEATURES,1.) | {'pose_ok':'1'},dict.fromkeys(FEATURES,3.) | {'pose_ok':'1'}]
    normalizer = fit_normalizer(train)
    assert normalizer['center']==[2.]*12
    heldout = dict.fromkeys(FEATURES,999.) | {'pose_ok':'0'}
    vector = encode_metrics(heldout,normalizer)
    assert vector.shape==(25,)
    assert normalizer['center']==[2.]*12
    assert vector[12]==1. and vector[0]==0. and vector[-1]==0.


def test_single_network_receives_metric_gradients_and_ablation_changes_outputs():
    torch.set_num_threads(2);torch.manual_seed(7)
    net = MetricFusionNet().eval()
    images = torch.randn(2,3,160,160)
    metrics = torch.randn(2,25,requires_grad=True)
    logits = net(images,metrics)
    assert logits.shape==(2,3)
    logits.square().sum().backward()
    assert metrics.grad.abs().sum()>0
    assert not torch.allclose(logits,net(images,metrics,ablation='no_metrics'))


def test_missing_body_or_multiple_people_remains_unknown():
    rows = [{'pose_ok':'0','quality_reason':'no_pose'},
            {'pose_ok':'0','quality_reason':'low_pose_quality'},
            {'pose_ok':'0','quality_reason':'multiple_people','extraction_route':'yolo_keypoints'}]
    assert [body_evidence(r) for r in rows]==[False,True,False]
    probability = np.array([[.01,.01,.98]]*3)
    assert predictions_for(rows,probability,.5).tolist()==['unknown','prone','unknown']


def test_fusion_training_export_and_safe_weight_loading(tmp_path,monkeypatch):
    # Tiny artificial images/metrics verify the training/export path only.
    rows=[]
    for group in range(20):
        for i,label in enumerate(LABELS):
            path=tmp_path/f'{group}-{label}.png'
            Image.new('RGB',(24,24),(40+i*60,40+group,50)).save(path)
            row=dict(image_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                posture=label,source_group=f'g-{group}',session_id='',baby_id='',reviewed='1',
                label_source='human_review',training_eligible='1',pose_ok='1',quality_reason='ok')
            row.update(dict.fromkeys(FEATURES,float(i+group*.01)))
            rows.append(row)
    features=tmp_path/'features.csv';write_csv(features,rows,list(rows[0]))
    output=tmp_path/'model'
    report=train_fusion(features,output,epochs=1,warmup=1,batch_size=12,pretrained=False)
    config=json.loads((output/'fusion.json').read_text())
    assert config['normalizer']['fitted_on']=='training_only'
    assert report['metric_input_count']==12
    assert set(report['ablations'])=={'no_metrics','no_image'}
    assert not report['research_95pct_point_target_met']
    assert hashlib.sha256((output/'posture_fusion.pt').read_bytes()).hexdigest()==report['model_sha256']
    state=torch.load(output/'posture_fusion.pt',weights_only=True,map_location='cpu')
    model=MetricFusionNet();model.load_state_dict(state,strict=True)
    split=read_csv(output/'split_manifest.csv')
    sets=[{r['source_group'] for r in split if r['split']==fold} for fold in ('train','validation','test')]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    from cribwatch import fusion
    from cribwatch.features import Observation
    class FakeVision:
        def observe(self,rgb):
            return Observation(dict.fromkeys(FEATURES,.5),True,'ok')
        def close(self):pass
    monkeypatch.setattr(fusion,'verify_classifier_assets',lambda *args:'pose_landmarker_lite.task')
    monkeypatch.setattr(fusion,'create_vision',lambda *args,**kwargs:FakeVision())
    predictor=fusion.FusionPredictor(output,models_dir=tmp_path,showcase=True)
    result=predictor.predict(np.zeros((160,160,3),np.uint8))
    assert len(result)==6 and set(result[2])==set(LABELS)
    predictor.close()
    with pytest.raises(ValueError,match='settings changed'):
        train_fusion(features,output,epochs=2,warmup=1,batch_size=12,pretrained=False)
    config['temperature']=config['temperature']+1.
    (output/'fusion.json').write_text(json.dumps(config,indent=2))
    with pytest.raises(ValueError,match='configuration'):
        fusion.FusionPredictor(output,models_dir=tmp_path,showcase=True)
