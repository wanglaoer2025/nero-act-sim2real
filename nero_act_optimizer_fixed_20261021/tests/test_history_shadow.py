import hashlib
import importlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from tests.test_visual_policy import config,statistics


def capture(tmp_path):
    path=tmp_path/'capture';path.mkdir();frames=[];receipts=[]
    for i in range(5):
        stamp=10+i*.05;image=np.full((90,160,3),i,np.uint8);name=f'{i}.png';cv2.imwrite(str(path/name),image)
        frames.append(dict(frame_id=i,exposure_s=stamp,feedback_s=stamp,received_s=stamp+.01,admitted_s=stamp+.01,
            state=[0]*7+[.074],geometry_features_m=[.3]*6,rgb_path=name,rgb_sha256=hashlib.sha256((path/name).read_bytes()).hexdigest()))
        receipts.append(dict(receipt_id=str(i),rows=[[0]*7+[.071]],sent_s=stamp-.02,ack_s=stamp-.01,
            accepted=True,kind='submitted_prefix',scope='s'))
    manifest=dict(schema='numeric_history_saved_observation_v1',source_scope='nominal_simulation',
        calibration_sha256='cal',camera_metadata_sha256='camera',session_id='s',epoch=1,after_s=9.9,
        frames=frames,receipts=receipts)
    (path/'manifest.json').write_text(json.dumps(manifest));return path


def api():
    assert importlib.util.find_spec('executor.history_shadow'), 'Offline saved-history worker missing'
    return importlib.import_module('executor.history_shadow')


def test_shadow_never_imports_robot_or_opens_camera(tmp_path):
    class NoHardware:
        def find_spec(self,fullname,*args):
            if fullname.split('.')[0] in {'rclpy','pyAgxArm','pyzed'}:raise AssertionError('Hardware import: '+fullname)
    guard=NoHardware();sys.meta_path.insert(0,guard)
    try:
        m=api();c=config();c.update(calibration_sha256='cal',numeric_history_profile='causal_numeric_gru5_v1')
        seen=[]
        def predict(rgb,state,previous,*,geometry_features_m,history):
            seen.append(history);return np.tile(state,(20,1))
        policy=SimpleNamespace(config=c,stats=statistics(c),metadata={'checkpoint_sha256':'fixture'},predict=predict)
        report=m.run_saved_shadow(policy,capture(tmp_path),tmp_path/'output')
        assert report['motion_sent'] is False and report['real_hardware_tested'] is False
        assert report['live_frontend_qualified'] is False
        assert len(seen)==1
        np.testing.assert_allclose(seen[0].observations[:,8],[-.3]*5,atol=1e-6)
        assert np.load(tmp_path/'output/raw20x8.npy').shape==(20,8)
    finally:sys.meta_path.remove(guard)


def test_saved_history_rejects_changed_rgb_and_path_escape(tmp_path):
    m=api();path=capture(tmp_path);c=config();c.update(calibration_sha256='cal')
    manifest=json.loads((path/'manifest.json').read_text())
    manifest['frames'][0]['rgb_path']='../outside.png';(path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):m.load_history_capture(path,statistics(c),c)


def test_pilot_export_cannot_claim_formal_qualification(tmp_path):
    from tests.test_history_training import output_fixture
    from training.history import finalize_history_training
    import torch
    checkpoint,protected=output_fixture(tmp_path)
    c=config();c.update(num_epochs=0,training_budget={'kind':'updates','limit':1000},training_integrity_required=True)
    torch.save(dict(config=c,model={},epoch=0),checkpoint)
    finalize_history_training(tmp_path,c,dict(updates=1000,epochs=0),protected)
    from training.export import export_model
    with pytest.raises(ValueError,match='pilot|budget'):
        export_model(checkpoint,tmp_path/'invalid_export')
    assert not (tmp_path/'invalid_export').exists()
