from types import SimpleNamespace as P

import numpy as np

from cribwatch.display import draw_overlay, make_output
from cribwatch.features import extract_metrics, observation_for_schema
from cribwatch.frame_roi import crop_frame, parse_roi
from cribwatch.tcp_alarm import TCPAlarmClient


def test_geometry_tracks_real_coordinates_and_does_not_draw_an_invented_skeleton():
    points = [P(x=.5, y=.2, visibility=.9, presence=.9) for _ in range(33)]
    for i, x, y in ((11,.2,.3),(12,.8,.3),(23,.3,.7),(24,.7,.7)):
        points[i] = P(x=x, y=y, visibility=.9, presence=.9)
    observation = extract_metrics(points, 640, 640)
    assert observation.geometry['points']['left_shoulder']['x'] == .2
    assert observation.geometry['points']['left_shoulder']['usable']
    rgb = np.zeros((480,640,3),np.uint8)
    output = make_output('prone',.8,'ok')
    output.update(geometry=observation.geometry,metrics=observation.metrics,bounding_box=observation.pose_box)
    frame = draw_overlay(rgb,output)
    assert np.array_equal(rgb,np.zeros_like(rgb))
    assert frame.shape[1] == 1010  # Measured frame plus metric panel.
    assert frame[145+round(.3*479),round(.5*639)].sum() > 0  # Shoulder vector location.
    points[11].visibility = .1
    poor = extract_metrics(points,640,640)
    assert not poor.geometry['points']['left_shoulder']['usable']
    assert poor.geometry['points']['left_shoulder']['confidence'] == .1


def test_normalized_roi_crops_pixels_without_transforming_source():
    rgb = np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3)
    selected = crop_frame(rgb,parse_roi('.25,.2,.75,.8'))
    assert np.array_equal(selected,rgb[20:80,50:150])
    assert not np.shares_memory(selected,rgb)


def test_large_geometry_stays_local_and_cannot_overflow_alarm_packet():
    messages = []
    client = TCPAlarmClient('unused',b'x'*32)
    def exchange(message):
        messages.append(message)
        return {'nonce':'n'*64} if message['op']=='challenge' else {'accepted':True}
    client.exchange = exchange
    status = make_output('prone',.8,'ok')
    status.update(geometry={'large':'x'*10000},metrics={'raw':'x'*10000})
    client.send(status)
    payload = messages[-1]['payload']
    assert 'geometry' not in payload and 'metrics' not in payload
    assert payload['posture']=='prone' and payload['nonce']=='n'*64


def test_directed_angles_keep_facing_information_and_ignore_in_plane_rotation():
    points = [P(x=.5,y=.2,visibility=.9,presence=.9) for _ in range(33)]
    for i,x,y in ((11,.2,.3),(12,.8,.3),(23,.3,.7),(24,.7,.7)):
        points[i]=P(x=x,y=y,visibility=.9,presence=.9)
    old = extract_metrics(points,640,640)
    directed = observation_for_schema(old,3)
    points[11],points[12] = points[12],points[11]
    points[23],points[24] = points[24],points[23]
    reversed_old = extract_metrics(points,640,640)
    reversed_new = observation_for_schema(reversed_old,3)
    assert old.metrics['shoulder_angle_deg']==reversed_old.metrics['shoulder_angle_deg']
    assert directed.metrics['shoulder_angle_deg']==-reversed_new.metrics['shoulder_angle_deg']
    assert abs(directed.metrics['shoulder_angle_deg'])==90.
    for point in points:
        point.x,point.y=1-point.y,point.x
    rotated=observation_for_schema(extract_metrics(points,640,640),3)
    assert abs(rotated.metrics['shoulder_angle_deg']-reversed_new.metrics['shoulder_angle_deg'])<1e-8
    assert old.metrics['shoulder_angle_deg']==0.  # Original contract remains unchanged.


def test_failed_inference_never_attaches_previous_geometry_to_new_frame():
    import threading,time
    from cribwatch.showcase import InferenceWorker
    worker=object.__new__(InferenceWorker)
    worker.stop=threading.Event();worker.lock=threading.Lock();worker.roi=None
    worker.result=None
    worker.camera=P(get=lambda:(1,time.monotonic(),np.zeros((24,24,3),np.uint8)))
    class Broken:
        last_geometry={'points':{'old':True}}
        def predict(self,rgb):
            worker.stop.set()
            raise RuntimeError('test fixture')
    worker.predictor=Broken()
    worker.run()
    assert worker.result[3][0]=='unknown'
    assert worker.result[5] is None


def test_doll_capture_records_real_session_without_inventing_baby_id(tmp_path,monkeypatch):
    import time,csv
    from cribwatch import capture
    from PIL import Image
    class Camera:
        def __init__(self,*args,**kwargs):self.n=0
        def get(self):
            self.n+=1
            return self.n,time.monotonic(),np.full((32,32,3),[255,0,0],np.uint8)
        def close(self):pass
    monkeypatch.setattr(capture,'LatestCamera',Camera)
    count=capture.capture(tmp_path,posture='prone',subject_id='doll01',session_id='actual-session',seconds=.03)
    assert count==1
    with (tmp_path/'metadata.csv').open(newline='') as f:row=next(csv.DictReader(f))
    assert row['baby_id']=='' and row['session_id']=='actual-session' and row['subject_kind']=='doll'
    with Image.open(tmp_path/row['relative_path']) as image:
        assert image.getpixel((0,0))[0]>240
