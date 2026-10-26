"""Post-stop evidence only: ZED images and passive ROS subscriptions, no SDK."""
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/post_stop_contact_readonly'

def main():
    from executor.camera import SceneCamera
    from executor.runtime import SceneObserver, json_write
    OUT.mkdir(exist_ok=False)
    tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    before = tx()
    report = dict(mode='post_stop_passive_observation', motion_sent=False,
                  sdk_instantiated=False, action_clients_created=False,
                  wall_s=time.time(), observations=[], camera_frames=[])
    camera = node = None
    try:
        scene = json.loads((ROOT/'model/current_run27/scene_config.json').read_text())
        sensor = SceneObserver(scene, OUT)
        camera = SceneCamera(OUT/'camera')
        for _ in range(8):
            details = camera.capture()
            report['camera_frames'].append(details)
            try:
                report['observations'].append(sensor.measure(details))
            except Exception as exc:
                report.setdefault('measurement_errors', []).append(str(exc))
            time.sleep(.25)
        camera.close()
        camera = None
        import rclpy
        from sensor_msgs.msg import JointState
        rclpy.init()
        node = rclpy.create_node('post_stop_passive_contact_observer')
        names = [f'joint{i}' for i in range(1, 8)] + ['gripper']
        samples = []
        def receive(msg):
            indices = {name: i for i, name in enumerate(msg.name)}
            if not all(n in indices for n in names):
                return
            samples.append(dict(receipt_monotonic_s=time.monotonic(),
                                position=[msg.position[indices[n]] for n in names],
                                effort=[msg.effort[indices[n]] for n in names]
                                if len(msg.effort) == len(msg.name) else []))
        sub = node.create_subscription(JointState, '/feedback/joint_states', receive, 10)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
        report['feedback_samples'] = samples
        report['gated_command_publishers'] = node.count_publishers('/control/gated_joint_states')
        report['command_driver_alive'] = Path('/proc/2301154').exists()
        report['gate_alive'] = Path('/proc/2301191').exists()
        report['runtime_alive'] = Path('/proc/2301192').exists()
        report['readonly_driver_pid'] = 2301664
        report['readonly_driver_cmdline'] = Path('/proc/2301664/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if camera is not None:
            camera.close()
        if node is not None:
            node.destroy_node()
            rclpy.try_shutdown()
        report.update(can_tx_before=before, can_tx_after=tx())
        report['no_can_tx_during_observation'] = before == report['can_tx_after']
        json_write(OUT/'report.json', report)
        print(json.dumps({k: report.get(k) for k in [
            'error', 'can_tx_before', 'can_tx_after', 'gated_command_publishers',
            'command_driver_alive', 'gate_alive', 'runtime_alive', 'readonly_driver_pid']}
        ), flush=True)
        print('feedback_samples', len(report.get('feedback_samples', [])), flush=True)

if __name__ == '__main__':
    main()
