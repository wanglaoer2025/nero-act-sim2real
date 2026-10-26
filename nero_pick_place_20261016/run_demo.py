"""Supervised historical-parameter grasp/place demo; default plans only.

Reuses the frozen teacher's geometry, health guards and staged execution. This
entry is a task demonstration, not teacher_session collection or model training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/home/user/zyh/data/nero-teacher/runtime-v4')


def check_model_tcp(config):
    offset = np.asarray(config.get('tcp_offset_flange_xyz_m'), float)
    if (config.get('approximate') is not True or offset.shape != (3,)
            or not np.isfinite(offset).all()
            or not np.allclose(offset, [.152, 0, -.0235], rtol=0, atol=1e-12)):
        raise ValueError('Only the explicitly accepted approximate model TCP is authorized')


def wait_for_control_graph(session, timeout_s=5):
    required = ('agx_arm_ctrl_single_node', 'control_gate', 'controller_manager',
                'arm_controller', 'gripper_controller', 'move_group')
    deadline = time.monotonic() + timeout_s
    while True:
        nodes = session.node.get_node_names_and_namespaces()
        counts = {name: sum(node_name == name for node_name, _ in nodes) for name in required}
        if any(count > 1 for count in counts.values()):
            raise RuntimeError(f'Duplicate ROS control nodes: {counts}')
        if all(count == 1 for count in counts.values()):
            return
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'ROS control graph discovery incomplete: {counts}')
        session.rclpy.spin_once(session.node, timeout_sec=min(.05, remaining))


def camera_health_barrier(camera, read_health, timeout_s=2):
    open_camera = camera.open

    def open_after_feedback_ready():
        open_camera()
        deadline, healthy = time.monotonic()+timeout_s, 0
        while time.monotonic() < deadline:
            try:
                read_health()
                healthy += 1
            except TimeoutError:
                healthy = 0
            if time.monotonic() >= deadline:
                break
            if healthy == 2:
                return
            time.sleep(.02)
        raise TimeoutError('Camera initialized but complete fresh feedback did not become ready')

    camera.open = open_after_feedback_ready


def authorize_stage(action, token, *, targets=None, now=None):
    print(f'\n现场确认：{action}；输入 {token} 继续，其他输入取消。', flush=True)
    try:
        if input().strip() != token:
            return False
        if targets is not None:
            from perception import require_current_targets
            require_current_targets(targets, time.time() if now is None else now)
        return True
    except EOFError:
        return False


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False,
                  default=lambda item: item.tolist() if isinstance(item, np.ndarray) else str(item))


def guard_carried_cup(session, plan, report, *, before_retreat=None, tcp_offset=(.135, 0, 0)):
    from cup_sweep import validate_cup_trajectory
    original = session.plan_to_pose
    stages = iter(('pregrasp_high_pose', 'grasp_pose', 'lift_pose',
                   'carry_pose', 'place_pose', 'retreat_pose'))

    def plan_to_pose(target, scale, *, start_state=None, description='目标位姿'):
        stage = next(stages)
        if not np.allclose(target, plan['flange'][stage], rtol=0, atol=1e-9):
            raise ValueError('Unexpected grasp/place stage target')
        if stage == 'retreat_pose' and before_retreat is not None:
            before_retreat()
        start = session.joint_state() if start_state is None else start_state
        if stage == 'retreat_pose' and before_retreat is not None:
            from moveit_msgs.srv import GetStateValidity
            client = session.node.create_client(GetStateValidity, '/check_state_validity')
            try:
                if not client.wait_for_service(timeout_sec=5):
                    raise TimeoutError('Retreat start collision diagnostic unavailable')
                request = GetStateValidity.Request()
                request.group_name = 'arm'
                request.robot_state.joint_state = start
                response = session._wait(client.call_async(request), '退让起点碰撞检查')
                report['retreat_start_collision_check'] = dict(
                    valid=response.valid,
                    contacts=[dict(body1=c.contact_body_1, body2=c.contact_body_2,
                                   penetration_m=c.depth) for c in response.contacts])
                if not response.valid:
                    raise RuntimeError(f'退让起点碰撞：{report["retreat_start_collision_check"]}')
            finally:
                session.node.destroy_client(client)
        trajectory = (session.plan_cartesian([target], scale, start_state=start, description=description)
                      if stage == 'retreat_pose' else
                      original(target, scale, start_state=start, description=description))
        if stage in ('lift_pose', 'carry_pose', 'place_pose'):
            evidence = validate_cup_trajectory(session, trajectory, start, tcp_offset=tcp_offset)
            report.setdefault('cup_sweep_checks', []).append(dict(stage=stage, **evidence))
            print('杯体扫掠通过：', stage, evidence, flush=True)
        return trajectory

    session.plan_to_pose = plan_to_pose


def close_gate_before_cancel(session, report):
    from std_srvs.srv import SetBool
    original = session.cancel
    client = session.node.create_client(SetBool, '/control_enable')

    def cancel():
        try:
            if not client.wait_for_service(timeout_sec=.2):
                raise TimeoutError('ROS control gate service unavailable')
            future = client.call_async(SetBool.Request(data=False))
            session.rclpy.spin_until_future_complete(session.node, future, timeout_sec=.5)
            if not future.done() or future.exception() or not future.result().success:
                raise RuntimeError('ROS gate did not acknowledge closure')
            report['gate_closed_for_cancel'] = True
        except Exception as exc:
            report['gate_close_error'] = str(exc)
        finally:
            original()

    session.cancel = cancel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--candidate-tcp', type=Path, help='explicitly selected model TCP')
    parser.add_argument('--accept-approximate-model-tcp', action='store_true')
    args = parser.parse_args()
    if args.execute and (args.candidate_tcp is None or not args.accept_approximate_model_tcp):
        raise PermissionError('Historical 135mm TCP has a confirmed retreat collision; '
                              'execution needs the accepted model TCP and explicit opt-in')
    args.targets = args.targets.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(RUNTIME), '/home/user/zyh', '/home/user/zyh/src']
    sys.path.append('/home/user/zyh/third_party/openpi/packages/openpi-client/src')
    from perception import CAMERA, TCP, require_current_targets
    from scripts import nero_methylene_teacher as teacher
    from scripts.nero_teacher import verify_runtime
    from src.utils.moveit import MoveItSession, beaker_collision_cylinders, joint_state_after_trajectory
    verify_runtime(RUNTIME)
    targets = json.loads(args.targets.read_text())
    require_current_targets(targets, time.time())
    for path, field in [(CAMERA, 'calibration_sha256'), (TCP, 'tcp_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != targets[field]:
            raise ValueError(f'Geometry source changed: {path}')
    if args.candidate_tcp is not None:
        TCP = args.candidate_tcp.resolve()
        if args.execute:
            check_model_tcp(json.loads(TCP.read_text()))
    task_args = argparse.Namespace(**targets['parameters'])
    teacher.validate_args(task_args)
    plan = teacher.transfer_plan(np.asarray(targets['plan']['source_bottom_mm']),
                                 np.asarray(targets['plan']['destination_mm']),
                                 task_args, json.loads(TCP.read_text()))
    report = dict(mode='execute' if args.execute else 'plan', motion_sent=False,
                  success=False, task_success='unreviewed', confirmations=[], stages=[])
    report['tcp_source'] = str(TCP)
    report['model_tcp_comparison_only'] = args.candidate_tcp is not None and not args.execute
    session = observer = recorder = None
    cylinders = []
    try:
        if args.execute:
            if not sys.stdin.isatty():
                raise PermissionError('Live per-stage operator confirmation needs an interactive terminal')
            print('完整杯到蓝卡抓放；历史桌高0mm；已接受模型TCP[152,0,-23.5]mm（未实测）。', flush=True)
            print('当前10月14日相机标定；退出保持笛卡尔垂直避碰。', flush=True)
            print('ROS唯一发送者75%/200Hz/smoothed；轨迹转场5%、精细1%；夹爪0.5..1N。', flush=True)
            print('确认你仍在实体急停旁、杯子为空、全路径无人无障碍且没有其他控制者。', flush=True)
            print('保留每步NEXT、对中后GRASP、杯底落桌后RELEASE；失败取消，无自动回位。', flush=True)
            if not authorize_stage('确认以上参数和现场状态，开始本次逐步完整抓放', 'PICK'):
                raise PermissionError('PICK not supplied')
            require_current_targets(targets, time.time())
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                Path(__file__), ROOT / 'perception.py', ROOT / 'cup_sweep.py', args.targets,
                Path(teacher.__file__), RUNTIME / 'snapshot.json', CAMERA, TCP,
                Path('/home/user/wq/nero_full_task_20261016/task_replay.py'),
            ]}
            write_new(args.output / 'authorization.json', dict(
                operator_confirmation='PICK', authorized_wall_s=time.time(),
                authorization_consumed=True, source_hashes=hashes,
                scope='one supervised complete yellow-cup to blue-card grasp/place',
            ))
            report['confirmations'].append(dict(token='PICK', wall_s=time.time()))
            sys.path.insert(0, '/home/user/wq/nero_full_task_20261016')
            from task_replay import observer as connect_observer
            observer = connect_observer()
            session = teacher.guarded_session_class(MoveItSession)(
                observer, link_name='link7', timeout_s=60)
            wait_for_control_graph(session)
            session.require_smoothed_driver()
            session.read_health()
        else:
            session = MoveItSession(link_name='link7', timeout_s=60)
        start = session.joint_state()
        report['actual_start'] = dict(zip(start.name, start.position))
        def add_placed_cup():
            nonlocal cylinders
            cylinders = beaker_collision_cylinders(
                [('teacher_destination', np.asarray(plan['destination_mm']))], 0, 95, 75, 5)
            session.update_beaker_collisions(cylinders, add=True)

        guard_carried_cup(session, plan, report, before_retreat=add_placed_cup,
                          tcp_offset=json.loads(TCP.read_text())['tcp_offset_flange_xyz_m'])
        if args.execute:
            close_gate_before_cancel(session, report)
        session.add_table_collision(np.array(task_args.workspace_mm).reshape(3, 2), 0)
        cylinders = beaker_collision_cylinders(
            [('teacher_source', np.asarray(plan['source_bottom_mm']))], 0, 95, 75, 5)
        session.update_beaker_collisions(cylinders, add=True)
        if not args.execute:
            for name in ('pregrasp_high_pose', 'grasp_pose', 'lift_pose',
                         'carry_pose', 'place_pose', 'retreat_pose'):
                if name in ('grasp_pose', 'retreat_pose'):
                    # Mirror the explicitly confirmed gripper opening preceding these stages.
                    positions = dict(zip(start.name, start.position))
                    positions['gripper'] = .095
                    start.position = [positions[joint] for joint in start.name]
                scale = .05 if name in ('pregrasp_high_pose', 'carry_pose') else .01
                trajectory = session.plan_to_pose(plan['flange'][name], scale,
                                                   start_state=start, description=name)
                report['stages'].append(dict(name=name, points=len(trajectory.points),
                    duration_s=trajectory.points[-1].time_from_start.sec +
                    trajectory.points[-1].time_from_start.nanosec / 1e9))
                start = joint_state_after_trajectory(start, trajectory)
                if name == 'pregrasp_high_pose':
                    session.update_beaker_collisions(cylinders, add=False)
                    cylinders = []
            report['success'] = True
            return 0

        from src.act.record_yolo_episode import YoloEpisodeRecorder
        from scripts.nero_teacher_runner import recorder_class
        from ultralytics import YOLO
        # Initialize inference before camera/health recording deadlines become active.
        live_model = YOLO(task_args.weights)
        cup_id = next(index for index, name in live_model.names.items() if name == 'cup')
        live_model.predict(np.zeros((360, 640, 3), np.uint8), classes=[cup_id],
                           conf=task_args.confidence, imgsz=960, verbose=False)
        session.read_health()
        import os
        os.chdir(RUNTIME)
        recorder = recorder_class(YoloEpisodeRecorder, profile='supervised_historical_pick_place_demo')(
            args.output / 'episode.hdf5', frequency=20, action_lead_ms=0,
            secondary_camera_device=None, metadata=dict(
                task_name='yellow_cup_to_blue_card', synthetic=False,
                historical_table_z_base_mm=0, tcp_approximate=True,
                geometry_source=str(args.targets), calibration_sha256=targets['calibration_sha256'],
                tcp_source=str(TCP), tcp_offset_flange_xyz_m=[.152, 0, -.0235],
                destination_method=targets['destination_method'],
            ))
        camera_health_barrier(recorder.camera, session.read_health)
        recorder.start(observer.sdk, observer.gripper, float(observer.read_health(require_enabled=True)['state'][7]))
        session.recorder = recorder
        def confirm(action, token):
            nonlocal cylinders
            needs_fresh_target = action in (
                '移动到杯子正上方', '垂直下降到杯子抓取高度', '确认两指位于杯壁两侧并闭爪')
            accepted = authorize_stage(action, token)
            if accepted:
                if needs_fresh_target:
                    if time.time()-targets['capture_wall_s'] > 90:
                        from perception import refresh_from_recorder
                        refreshed = refresh_from_recorder(
                            recorder, task_args, targets,
                            args.output / f'revalidation_{len(report["confirmations"]):02d}',
                            model=live_model)
                        targets.update(refreshed)
                    require_current_targets(targets, time.time())
                # Pregrasp has physically completed; contact descent follows opening.
                if action == '在杯子上方打开夹爪' and cylinders:
                    session.update_beaker_collisions(cylinders, add=False)
                    cylinders = []
                report['confirmations'].append(dict(action=action, token=token, wall_s=time.time()))
                write_new(args.output / f'confirmation_{len(report["confirmations"]):02d}.json',
                          report['confirmations'][-1])
            return accepted

        report['motion_sent'] = True
        teacher.execute_transfer(session, recorder, plan, task_args, confirm=confirm)
        report['final_health'] = session.read_health()
        session.recorder = None
        recorder.finish(save=True)
        recorder = None
        report.update(success=True, task_success='awaiting_visual_review')
        return 0
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if recorder is not None and recorder.error is not None:
            report['recording_error'] = f'{type(recorder.error).__name__}: {recorder.error}'
        if session is not None and args.execute:
            try:
                session.cancel()
                report['cancel_requested'] = True
            except Exception as cancel_exc:
                report['cancel_error'] = str(cancel_exc)
        print(report['error'], flush=True)
        return 1
    finally:
        if session is not None and cylinders:
            try:
                session.update_beaker_collisions(cylinders, add=False)
            except Exception as exc:
                report['scene_cleanup_error'] = str(exc)
        try:
            teacher.close_resources(session, observer, recorder)
        finally:
            report['finished_wall_s'] = time.time()
            write_new(args.output / 'result.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
