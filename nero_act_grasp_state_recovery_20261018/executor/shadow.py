"""Real feedback plus live-camera ACT shadow. This entry has no execution mode."""
import argparse
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

import numpy as np

from executor.contract import FullTaskContract

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--queries', type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.queries <= 20:
        parser.error('queries must be in1..20')
    args.model, args.output = args.model.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    sys.path.insert(0, str(ROOT.parent/'nero_readonly_handoff_20261014'))
    from capture_nero_observations import ordered_feedback
    lock, latest, trace, errors = threading.Lock(), {}, [], []
    rclpy.init(args=[])
    node = rclpy.create_node('nero_full_act_readonly_shadow')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def feedback(message):
        try:
            state = ordered_feedback(message.name, message.position)
            item = dict(state=state.tolist(), receipt=time.monotonic(),
                        ros_stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9)
            with lock:
                if not latest or item['ros_stamp'] > latest['ros_stamp']:
                    latest['last_advance'] = item['receipt']
                latest.update(item)
                trace.append(item)
        except Exception as exc:
            errors.append(str(exc))
    node.create_subscription(JointState, '/feedback/joint_states', feedback, qos_profile_sensor_data)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    log = (args.output/'worker.log').open('x')
    worker = subprocess.Popen([sys.executable, '-m', 'executor.policy_worker', '--model', str(args.model),
        '--stream', '--output', str(args.output/'policy_inputs')], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    records, failure, initial_reference = [], None, None
    def receive(timeout):
        if not select.select([worker.stdout], [], [], timeout)[0]:
            raise TimeoutError('Read-only inference worker timed out')
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError('Inference worker exited; see worker.log')
        return json.loads(line)
    try:
        ready = receive(45)
        if ready.get('type') != 'ready':
            raise RuntimeError('Unexpected inference handshake')
        config = json.loads((args.model/'training_config.json').read_text())
        guard = FullTaskContract(config)
        for index in range(args.queries):
            with lock:
                item = dict(latest)
            now = time.monotonic()
            if (errors or not item or now-item['receipt'] > .2
                    or now-item.get('last_advance', 0) > .2):
                raise RuntimeError('Missing, frozen or stale actual feedback')
            ros_age = node.get_clock().now().nanoseconds*1e-9-item['ros_stamp']
            if not -.05 <= ros_age <= .25:
                raise RuntimeError('Invalid feedback ROS timestamp age')
            if initial_reference is None:initial_reference=float(item['state'][7])
            worker.stdin.write(json.dumps(dict(state=item['state'], observed_monotonic_s=item['receipt'],
                previous_gripper_target_m=initial_reference))+'\n')
            worker.stdin.flush()
            prediction = receive(10)
            raw = np.asarray(prediction['chunk'])
            age = time.monotonic()-item['receipt']
            issue = None
            try:
                guard.validate_prefix(raw, dict(state=item['state'], age_s=age), item['state'])
            except ValueError as exc:
                issue = str(exc)
            records.append(dict(index=index, actual_feedback=item, prediction=prediction,
                                observation_age_at_response_s=age, prefix_contract_issue=issue))
            print(json.dumps(dict(query=index, observation_age_s=age, prefix_issue=issue)), flush=True)
            time.sleep(.5)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if worker.poll() is None:
            try:
                worker.stdin.write('{"stop":true}\n')
                worker.stdin.flush()
                worker.wait(timeout=3)
            except Exception:
                worker.kill()
                worker.wait()
        log.close()
        executor.shutdown(timeout_sec=2)
        thread.join(timeout=2)
        node.destroy_node()
        rclpy.shutdown()
        (args.output/'actual_feedback.json').write_text(json.dumps(trace))
        (args.output/'predictions.json').write_text(json.dumps(records, indent=2))
        (args.output/'report.json').write_text(json.dumps(dict(
            mode='live_readonly_ACT_shadow', completed_queries=len(records),
            requested_queries=args.queries, success=failure is None and len(records) == args.queries,
            error=failure, motion_sent=False, gripper_sent=False, enable_sent=False,
            execution_ready=False, geometry_acceptance_performed=False,
            command_context_source='planned initial hold reference for readonly preview; no real command history claimed',
            note='Live prediction and prefix checks only; no task or deployment acceptance'), indent=2))
    return int(failure is not None)


if __name__ == '__main__':
    raise SystemExit(main())
