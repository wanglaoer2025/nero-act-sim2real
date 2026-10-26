"""One receive-only diagnostic: warm kernels, then admit ten NEW camera frames.

No publisher, service client, robot SDK, CAN socket, or command transport is used.
This is not a qualified visual executor or an authorization to move hardware.
"""
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl
import rclpy
from sensor_msgs.msg import JointState
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checked_runtime import load_projected
from executor.contract import FullTaskContract
from executor.model import make_policy_observation, physical_actions
from executor.ros_execution import RobotGeometry
from src.act.nero_dataset import preprocess_images


def main():
    out = ROOT/'evidence/live_readonly_preview_03_warm'
    out.mkdir(exist_ok=False)
    tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    before = tx()
    report = dict(mode='warmed_current_live_15D_ACT_readonly_diagnostic',
        records=[], warmup_records=[], motion_sent=False, controller_reset_sent=False,
        qualified_live_shadow=False, hardware_deployment_allowed=False,
        geometry_independently_calibrated=False, clock_exposure_verified=False,
        context_source='hypothetical measured-width hold; no submitted target history',
        perception_and_ACT_processes_isolated=True, can_tx_before=before, error=None)
    camera, worker, log, node, thread = None, None, None, None, None
    stopping = threading.Event()
    latest, lock = {}, threading.Lock()
    try:
        torch.set_num_threads(2)
        policy, stats = load_projected()
        scene_path = ROOT/'results/unqualified_nominal_full_development/scene_configuration.json'
        scene = json.loads(scene_path.read_text())
        robot = RobotGeometry(scene)
        contract = FullTaskContract(policy.config)
        source = ROOT/'evidence/live_readonly_preview_02/perception_worker_source.py'
        log = (out/'perception_worker.log').open('x')
        worker = subprocess.Popen(['/home/user/nero/bin/python', '-u', str(source),
            str(scene_path), str(out)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log, text=True, bufsize=1)

        def receive():
            if not select.select([worker.stdout], [], [], 45)[0]:
                raise TimeoutError('Perception response timeout')
            line = worker.stdout.readline()
            if not line:
                raise RuntimeError('Perception worker exited')
            result = json.loads(line)
            if result.get('type') == 'error':
                raise RuntimeError(result['error'])
            return result

        def measure(index, path, stamp):
            worker.stdin.write(json.dumps(dict(index=index, camera_frame=str(path),
                camera_received_monotonic_s=stamp))+'\n')
            worker.stdin.flush()
            value = receive()
            if (value.get('type') != 'geometry' or value.get('index') != index
                    or value.get('frame') != str(path)
                    or value.get('camera_monotonic_s') != stamp):
                raise RuntimeError('Frame identity mismatch')
            return value

        if receive() != {'type': 'ready'}:
            raise RuntimeError('Perception handshake failed')

        def predict(rgb, state, geometry):
            tcp = robot.pose(state)['tcp']
            features = np.r_[np.asarray(geometry['cup'])-tcp, np.asarray(geometry['target'])-tcp]
            current = make_policy_observation(state, stats, policy.config, float(state[7]),
                geometry_features_m=features)
            with torch.inference_mode():
                raw = physical_actions(policy(torch.from_numpy(current)[None].cuda(),
                    preprocess_images(rgb[None], [180, 320])[None].cuda()),
                    stats, bounded_gripper=True)[0].cpu().numpy()
            if raw.shape != (20, 8) or not np.isfinite(raw).all():
                raise RuntimeError('Invalid ACT prediction')
            return raw, features

        # Saved inputs ONLY initialize kernels. None enters the live records below.
        old = json.loads((ROOT/'evidence/live_readonly_preview_02/report.json').read_text())['records'][0]
        rgb = cv2.cvtColor(cv2.imread(old['camera_frame']), cv2.COLOR_BGR2RGB)
        state = np.asarray(old['feedback_state'], np.float32)
        for index in range(3):
            start = time.monotonic()
            geometry = measure(-index-1, Path(old['camera_frame']), start)
            predict(rgb, state, geometry)
            report['warmup_records'].append(dict(index=index, used_for_live_admission=False,
                duration_ms=(time.monotonic()-start)*1000))

        rclpy.init(args=[])
        node = rclpy.create_node('act_transfer_warmed_readonly_diagnostic')
        names = [f'joint{i}' for i in range(1, 8)]+['gripper']

        def feedback(message):
            if any(message.name.count(name) != 1 for name in names):
                return
            values = [message.position[message.name.index(name)] for name in names]
            with lock:
                latest.update(state=values, receipt=time.monotonic())

        node.create_subscription(JointState, '/feedback/joint_states', feedback, 10)
        def spin():
            while not stopping.is_set():
                rclpy.spin_once(node, timeout_sec=.01)
        thread = threading.Thread(target=spin)
        thread.start()
        deadline = time.monotonic()+5
        while not latest:
            if time.monotonic() >= deadline:
                raise RuntimeError('No current joint feedback')
            stopping.wait(.01)

        camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution, init.camera_fps = sl.RESOLUTION.HD720, 30
        init.depth_mode = sl.DEPTH_MODE.NONE
        if camera.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('Cannot open ZED')
        mat, runtime = sl.Mat(), sl.RuntimeParameters()
        for index in range(10):
            if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('No fresh camera frame')
            if camera.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, sl.Resolution(640, 360)) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('Camera retrieve failed')
            stamp = time.monotonic()
            bgr = np.asarray(mat.get_data()).copy()[::-1, ::-1, :3]
            with lock:
                paired = dict(latest)
            if abs(paired['receipt']-stamp) > .05:
                raise RuntimeError('Feedback and camera receipt differ by more than50ms')
            state = np.asarray(paired['state'], np.float32)
            path = out/f'camera_{index:03d}.png'
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError('Cannot save current frame')
            geometry = measure(index, path, stamp)
            raw, features = predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), state, geometry)
            age = time.monotonic()-stamp
            issue = None
            try:
                contract.validate_prefix(raw, dict(state=state, age_s=age), state)
            except ValueError as exc:
                issue = str(exc)
            report['records'].append(dict(index=index, camera_frame=str(path),
                camera_received_monotonic_s=stamp, feedback_received_monotonic_s=paired['receipt'],
                feedback_state=state.tolist(), cup_bottom_m=geometry['cup'], target_m=geometry['target'],
                geometry_relative_m=features.tolist(), end_to_end_ms=age*1000,
                prefix_contract_issue=issue, raw_first_two=raw[:2].tolist()))
            if tx() != before:
                raise RuntimeError('CAN TX changed during receive-only diagnostic')
    except Exception as exc:
        report['error'] = repr(exc)
    finally:
        if camera is not None:
            camera.close()
        stopping.set()
        if thread is not None:
            thread.join(timeout=2)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if worker is not None and worker.poll() is None:
            try:
                worker.stdin.write('{"stop":true}\n')
                worker.stdin.flush()
                worker.wait(timeout=3)
            except Exception:
                worker.terminate()
                worker.wait(timeout=3)
        if log is not None:
            log.close()
        records = report['records']
        report.update(completed_queries=len(records), finite_predictions=len(records),
            prefix_contract_passes=sum(r['prefix_contract_issue'] is None for r in records),
            end_to_end_ms_range=([min(r['end_to_end_ms'] for r in records),
                max(r['end_to_end_ms'] for r in records)] if records else None), can_tx_after=tx())
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ['records','warmup_records']}, indent=2))
    if report['error'] is not None or tx() != before:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
