import json,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

class ReplayStartupTests(unittest.TestCase):
    def test_camera_and_perception_warm_before_motion_watchdog_then_enable_for_first_phase(self):
        from executor.replay import execute
        events=[];holder={}
        class Session:
            def __init__(self,*args):
                holder['session']=self;self.monitoring=False;self.trace=[];self.health_trace=[]
                self.node=types.SimpleNamespace(destroy_node=lambda:None)
                self.ros=types.SimpleNamespace(try_shutdown=lambda:None)
                self.geometry=types.SimpleNamespace(pose=lambda _:dict(tcp=np.array([0,0,.255])))
            def inventory(self):return [0]*8
            def ready(self,_):pass
            def warm_services(self):events.append('services')
            def enable_monitoring(self):self.monitoring=True;events.append('monitoring')
            def spin(self,_):pass
            def configure_initial_scene(self):events.append('scene')
            def clients(self):return []
            def actual(self):return [0]*8
            def run_phase(self,*args):
                events.append('phase');assert self.monitoring
                raise RuntimeError('EXPECTED_NO_ROBOT_PHASE')
            def cancel_all(self):events.append('cancel')
            def contact_permission(self,_):pass
        class Camera:
            def __init__(self,*args):
                events.append('camera');assert 'session' not in holder,'camera warmed after robot receiver creation'
                time.sleep(.01)
            def capture(self):return {}
            def close(self):pass
        def sensor(*args):
            events.append('perception');assert 'session' not in holder,'perception warmed after robot receiver creation'
            return types.SimpleNamespace(measure=lambda frame:{})
        module=types.ModuleType('task_replay');module.observer=lambda:types.SimpleNamespace(disconnect=lambda:None)
        scene=dict(pick_xy_m=[0,0],place_xy_m=[.2,0])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);model=out/'model';model.mkdir();(model/'scene_config.json').write_text(json.dumps(scene))
            geometry=out/'geometry.json';geometry.write_text('{}')
            args=types.SimpleNamespace(model=model,geometry=geometry,output=out)
            measured=[dict(cup_xyz=np.array([0,0,0]),target_xyz=np.array([.2,0,0]))]
            with patch.dict(sys.modules,{'task_replay':module}),patch('executor.replay_session.replay_session_class',return_value=Session),patch('executor.camera.SceneCamera',Camera),patch('executor.replay.SceneObserver',side_effect=sensor),patch('executor.replay.observe',return_value=measured),patch('executor.replay.execution_scene',return_value=scene),patch('executor.replay.require_current_scene'),patch('executor.replay.require_approach_clearance'):
                report=execute(args,{},dict(source_rollout_sha256='trace',phases=[dict(name='approach',start=0,end=2)]),np.zeros((2,8)))
            self.assertIn('EXPECTED_NO_ROBOT_PHASE',report['error'])
            self.assertLess(events.index('camera'),events.index('monitoring'))
            self.assertLess(events.index('perception'),events.index('monitoring'))
            self.assertLess(events.index('monitoring'),events.index('phase'))
