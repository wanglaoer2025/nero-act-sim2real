import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parent


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        path=getattr(self,'entry_override',ROOT/'entry.py')
        spec=importlib.util.spec_from_file_location('_demo_review_entry',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)

    def test_frozen_or_pre_send_feedback_never_confirms_arrival(self):
        self.assertTrue(hasattr(self.module,'ArrivalEvidence'),'Missing post-send advancing arrival evidence')
        target=np.zeros(8);evidence=self.module.ArrivalEvidence(target,100.)
        sample=dict(state=target.tolist(),feedback=dict(motion_status=0),feedback_timestamps=[99.999]*19)
        for now in [0,.05,.10,.14]:self.assertFalse(evidence.update(sample,now))
        sample['feedback_timestamps']=[100.01]*19
        self.assertFalse(evidence.update(sample,.15))
        self.assertFalse(evidence.update(sample,.30))

    def test_distinct_stationary_post_send_feedback_can_confirm_arrival(self):
        self.assertTrue(hasattr(self.module,'ArrivalEvidence'),'Missing post-send advancing arrival evidence')
        target=np.zeros(8);evidence=self.module.ArrivalEvidence(target,100.)
        sample=dict(state=target.tolist(),feedback=dict(motion_status=0),feedback_timestamps=[100.01]*19)
        self.assertFalse(evidence.update(sample,1.))
        sample['feedback_timestamps']=[100.07]*19
        self.assertFalse(evidence.update(sample,1.06))
        sample['feedback_timestamps']=[100.13]*19
        self.assertTrue(evidence.update(sample,1.12))

    def test_one_frozen_joint_feedback_group_cannot_confirm_arrival(self):
        self.assertTrue(hasattr(self.module,'ArrivalEvidence'),'Missing post-send advancing arrival evidence')
        target=np.zeros(8);evidence=self.module.ArrivalEvidence(target,100.)
        sample=dict(state=target.tolist(),feedback=dict(motion_status=0),feedback_timestamps=[100.01]*19)
        self.assertFalse(evidence.update(sample,1.))
        sample['feedback_timestamps']=[100.13]*19;sample['feedback_timestamps'][0]=100.01
        self.assertFalse(evidence.update(sample,1.12))

    def test_missing_or_mismatched_hardware_dependency_binding_is_rejected_offline(self):
        self.assertTrue(hasattr(self.module,'hardware_source_paths'),'Missing hardware dependency binding')
        plan=json.loads((ROOT/'preparation/plan.json').read_text());rows=np.load(plan['targets_path'],allow_pickle=False)
        mandatory=self.module.hardware_source_paths()
        self.assertTrue(any(str(p).endswith('nero_pi05/robot.py') for p in mandatory))
        for path in mandatory:self.assertEqual(plan['source_sha256'].get(str(path)),hashlib.sha256(path.read_bytes()).hexdigest())
        bad=copy.deepcopy(plan);del bad['source_sha256'][str(mandatory[0])]
        with self.assertRaisesRegex(ValueError,'hardware.*binding'):self.module.validate_plan(bad,rows)
        bad=copy.deepcopy(plan);bad['source_sha256'][str(mandatory[0])]='0'*64
        with self.assertRaises(ValueError):self.module.validate_plan(bad,rows)

    def test_actual_receiver_never_acks_reused_pre_command_feedback(self):
        plan=json.loads((ROOT/'preparation/plan.json').read_text());rows=np.load(plan['targets_path'],allow_pickle=False)
        method=next(n for n in ast.walk(ast.parse(Path(self.module.__file__).read_text()))
            if isinstance(n,ast.FunctionDef) and n.name=='receive')
        value=np.r_[(rows[2,:7]+rows[3,:7])/2,rows[3,7]]
        sample=dict(state=value.tolist(),feedback=dict(motion_status=0),feedback_timestamps=[999.999]*19)
        clock=[0.]
        def monotonic():clock[0]+=.02;return clock[0]
        report=dict(events=[],arm_attempts=0,arm_returned=0);acks=[];sent=[];stopped=[]
        namespace=dict(np=np,time=SimpleNamespace(time=lambda:1000.,monotonic=monotonic,sleep=lambda _:None),
            rows=rows,plan=plan,digest='bound',report=report,dump=lambda *args:None,
            args=SimpleNamespace(output=Path('/unused')),Int32=lambda **kw:kw,
            ArrivalEvidence=getattr(self.module,'ArrivalEvidence',None))
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual receive>','exec'),namespace)
        receiver=SimpleNamespace(index=3,failure=None,previous=rows[2],health=lambda _:sample,
            robot=SimpleNamespace(sdk=SimpleNamespace(move_j=lambda q:sent.append(q)),stop_sending=lambda:stopped.append(True)),
            ack=SimpleNamespace(publish=lambda message:acks.append(message)))
        message=SimpleNamespace(name=[f'joint{i}' for i in range(1,8)],position=rows[3,:7],
            header=SimpleNamespace(frame_id='bound:3',stamp=SimpleNamespace(sec=1000,nanosec=0)))
        namespace['receive'](receiver,message)
        self.assertEqual(acks,[],'Frozen pre-send feedback must not advance the actual ROS receiver')
        self.assertIsInstance(receiver.failure,TimeoutError)
        self.assertEqual(len(sent),1)


if __name__=='__main__':unittest.main()
