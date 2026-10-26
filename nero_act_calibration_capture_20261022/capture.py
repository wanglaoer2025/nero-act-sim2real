"""Passive mounted-board capture. No enable, mode, target or calibration publishing."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent


def flange_transform(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('Finite SDK flange pose6 in meters/radians required')
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix


def make_sample(window, after, image_wall_s, corners, K, *, role):
    if role not in ('fit', 'pose_check') or len(window) < 3:
        raise ValueError('Fit/pose-check role and stationary feedback window required')
    stamps = np.asarray([h['monotonic'] for h in window]+[after['monotonic']], float)
    if (not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0)
            or stamps[-2]-stamps[0] < .5 or stamps[-1]-stamps[-2] > .15):
        raise ValueError('At least500ms stationary window and fresh after-image feedback required')
    health = [*window, after]
    for item in health:
        status = item['feedback']
        if (status.get('arm_status') not in (0, 2) or status.get('err_code') != 0
                or status.get('ctrl_mode') != 1 or status.get('motion_status') != 0):
            raise ValueError('Controller unhealthy, moving, or not in CAN feedback mode')
    transforms = [flange_transform(item['flange_pose']) for item in health]
    first = transforms[0]
    if any(np.linalg.norm(t[:3, 3]-first[:3, 3]) > .0005
           or Rotation.from_matrix(first[:3, :3].T@t[:3, :3]).magnitude() > np.deg2rad(.1)
           for t in transforms[1:]):
        raise ValueError('Flange moved over0.5mm/0.1degree during the stationary window')
    sdk_stamps = np.asarray(window[-1]['feedback_timestamps']+after['feedback_timestamps'], float)
    if (sdk_stamps.shape != (38,) or not np.isfinite(np.r_[sdk_stamps, image_wall_s]).all()
            or np.max(np.abs(sdk_stamps-image_wall_s)) > .05):
        raise ValueError('Camera/SDK native timestamps differ by more than50ms')
    pixels, K = np.asarray(corners, float), np.asarray(K, float)
    if (pixels.shape != (4, 2) or K.shape != (3, 3) or not np.isfinite(np.r_[pixels.ravel(), K.ravel()]).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or abs(cv2.contourArea(pixels.astype(np.float32))) < 16):
        raise ValueError('Exactly one complete finite marker and rectified intrinsics required')
    h = .095/2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    found = cv2.solvePnPGeneric(obj, pixels, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if found[0]:
        for rv, tv in zip(found[1], found[2]):
            R, t = cv2.Rodrigues(rv)[0], np.asarray(tv).reshape(3)
            if not np.isfinite(np.r_[R.ravel(), t]).all() or np.any((obj@R.T+t)[:, 2] <= 0):
                continue
            projected = cv2.projectPoints(obj, rv, tv, K, None)[0].reshape(4, 2)
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
            candidates.append(dict(transform=T.tolist(), rmse_px=float(np.sqrt(np.mean(np.sum((projected-pixels)**2, axis=1))))))
    candidates.sort(key=lambda c: c['rmse_px'])
    if not candidates or candidates[0]['rmse_px'] > 1.:
        raise ValueError('Marker positive-depth/reprojection check failed')
    if len(candidates) > 1 and candidates[1]['rmse_px'] <= 1.:
        a, b = [np.asarray(c['transform']) for c in candidates[:2]]
        if (np.linalg.norm(a[:3, 3]-b[:3, 3]) > .001
                or Rotation.from_matrix(a[:3, :3].T@b[:3, :3]).magnitude() > np.deg2rad(1)):
            raise ValueError('Planar marker pose ambiguous; use a more oblique viewing angle')
    return dict(role=role, marker_dictionary='DICT_4X4_50', marker_id=0, marker_black_side_m=.095,
        pixel_space='rectified_left', corners_optical=pixels.tolist(), K=K.tolist(),
        transform_base_from_flange_m=transforms[-1].tolist(),
        transform_camera_from_marker_m=candidates[0]['transform'], marker_pose_candidates=candidates,
        stationary_health=window, health_after=after, image_wall_s=float(image_wall_s),
        independent_absolute_validation=False, exposure_clock_verified=False,
        calibration_verified=False, motion_sent=False)


def dump(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False,
                      default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def require_distinct_pose(sample, previous_events):
    current = np.asarray(sample['transform_base_from_flange_m'])
    for event in previous_events:
        if not event['accepted']: continue
        previous = np.asarray(event['transform_base_from_flange_m'])
        if (np.linalg.norm(current[:3, 3]-previous[:3, 3]) < .001
                and Rotation.from_matrix(previous[:3, :3].T@current[:3, :3]).magnitude() < np.deg2rad(1)):
            raise ValueError('Duplicate flange pose within1mm and1degree, including across fit/check roles')


def camera_metadata(profile, serial, K):
    if str(serial) != str(profile['camera_serial']):
        raise ValueError('Camera serial differs from measured marker profile')
    return dict(serial=str(serial), image_size_wh=[640, 360], pixel_space='rectified_left',
                image_rotation_degrees=0, preview_rotation_degrees=180, K=np.asarray(K).tolist())


def marker_pixels(corners, ids):
    if ids is None: return None
    labels = np.asarray(ids).ravel()
    requested = np.flatnonzero(labels == 0)
    if len(labels) != len(corners) or len(requested) != 1: return None
    return np.asarray(corners[int(requested[0])]).reshape(4, 2)


def record_attempt(output, report, role, acquire, tx):
    folder = output/f'attempt_{len(report["events"]):03d}'
    folder.mkdir(exist_ok=False)
    event = dict(role=role, directory=str(folder), accepted=False, state='pending')
    payload = dict(event=event, sample=None, health_window=[], health_after=None,
                   camera_metadata=report['camera_metadata'])
    report['events'].append(event)
    dump(folder/'sample.json', payload)
    dump(output/'manifest.json', report)
    interrupted = None
    image_attempted = False
    try:
        acquire(payload)
        require_distinct_pose(payload['sample'], report['events'])
        image_attempted = True
        if payload.get('native') is None or not cv2.imwrite(str(folder/'optical.png'), payload['native']):
            raise OSError('Cannot save actual camera frame')
        event.update(accepted=True, state='accepted',
                     transform_base_from_flange_m=payload['sample']['transform_base_from_flange_m'])
        payload['optical_sha256'] = hashlib.sha256((folder/'optical.png').read_bytes()).hexdigest()
    except BaseException as exc:
        event.update(accepted=False, state='rejected' if isinstance(exc, Exception) else 'interrupted', error=repr(exc))
        if not isinstance(exc, Exception): interrupted = exc
    finally:
        native = payload.pop('native', None)
        if native is not None and not image_attempted:
            try:
                if not cv2.imwrite(str(folder/'optical.png'), native): raise OSError('Cannot save diagnostic image')
            except BaseException as exc:
                event['image_error'] = repr(exc)
                if not isinstance(exc, Exception): interrupted = exc
        try:
            dump(folder/'sample.json', payload)
        except BaseException as exc:
            event.update(accepted=False, state='persistence_failed', error=repr(exc))
            raise
        finally:
            counts = Counter(e['role'] for e in report['events'] if e['accepted'])
            report.update(fit_samples=counts['fit'], pose_check_samples=counts['pose_check'])
            try: report['can_tx_latest'] = tx()
            except Exception as exc: report['can_counter_error'] = repr(exc)
            dump(output/'manifest.json', report)
    print(json.dumps(event), flush=True)
    if interrupted is not None: raise interrupted
    return 'SAVED '+role if event['accepted'] else 'REJECTED: '+event.get('error', 'unknown')


def install_cleanup_signals():
    def stop(signum, _):
        raise KeyboardInterrupt('Capture stopped by signal '+str(signum))
    for signum in (signal.SIGTERM, signal.SIGHUP): signal.signal(signum, stop)


def finalize_session(output, report, cam, robot, gui_close, tx):
    errors = []
    for label, cleanup in [('camera', cam.close if cam is not None else None),
                           ('robot', robot.disconnect if robot is not None else None), ('gui', gui_close)]:
        if cleanup is None: continue
        try: cleanup()
        except BaseException as exc: errors.append(dict(component=label, error=repr(exc)))
    report.update(running=False, ended_local=datetime.datetime.now().astimezone().isoformat(), can_tx_end=None)
    try: report['can_tx_end'] = tx()
    except BaseException as exc: errors.append(dict(component='CAN counter', error=repr(exc)))
    report['cleanup_errors'] = errors
    dump(output/'manifest.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--once', action='store_true', help='One passive capture diagnostic without a window')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, '/home/user/wq/nero_act_transfer_resume_20261021')
    from robot_health import observer
    import pyzed.sl as sl
    profile_path = ROOT/'marker_measured_95mm/marker_profile.json'
    profile = json.loads(profile_path.read_text())
    if profile['black_square_size_m'] != .095 or profile['dictionary'] != 'DICT_4X4_50' or profile['id'] != 0:
        raise ValueError('Current operator-measured95mm ID0 profile required')
    tx = lambda: int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    report = dict(started_local=datetime.datetime.now().astimezone().isoformat(), running=True,
        marker_profile=profile, events=[], fit_samples=0, pose_check_samples=0,
        sdk_send_blocked=True, motion_sent=False, controller_reset_sent=False,
        calibration_verified=False, hardware_deployment_allowed=False, can_tx_start=None,
        manual_operator_motion_is_external=True)
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), profile_path, Path('/home/user/wq/nero_act_transfer_resume_20261021/robot_health.py')]}
    robot, cam = None, None
    title = 'NERO calibration READ ONLY | S fit | H pose check | Q close'
    dump(args.output/'manifest.json', report)
    install_cleanup_signals()
    try:
        report['can_tx_start'] = tx()
        robot = observer()
        cam = sl.Camera();init = sl.InitParameters()
        init.camera_resolution, init.camera_fps, init.depth_mode = sl.RESOLUTION.HD720, 30, sl.DEPTH_MODE.NONE
        init.set_from_serial_number(int(profile['camera_serial']))
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError('ZED open failed')
        res, mat, runtime = sl.Resolution(640, 360), sl.Mat(), sl.RuntimeParameters()
        info = cam.get_camera_information(res); intr = info.camera_configuration.calibration_parameters.left_cam
        K = np.array([[intr.fx, 0., intr.cx], [0., intr.fy, intr.cy], [0., 0., 1.]])
        report['camera_metadata'] = camera_metadata(profile, info.serial_number, K)
        params = cv2.aruco.DetectorParameters();params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)

        def image():
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('ZED grab failed')
            if cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, res) != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError('ZED retrieve failed')
            return np.asarray(mat.get_data()).copy()[..., :3], cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()

        def save_capture(role):
            def acquire(payload):
                window = payload['health_window']
                start = time.monotonic()
                while time.monotonic()-start < .65:
                    window.append(robot.read_health(require_enabled=True)); time.sleep(.05)
                window.append(robot.read_health(require_enabled=True))
                native, stamp_ns = image()
                payload['native'] = native
                after = robot.read_health(require_enabled=True)
                payload['health_after'] = after
                corners, ids, _ = detector.detectMarkers(native)
                pixels = marker_pixels(corners, ids)
                intr_now = cam.get_camera_information(res).camera_configuration.calibration_parameters.left_cam
                if not np.array_equal(K, [[intr_now.fx, 0., intr_now.cx], [0., intr_now.fy, intr_now.cy], [0., 0., 1.]]):
                    raise ValueError('Camera intrinsics changed during session; start a new calibration capture')
                payload['sample'] = make_sample(window, after, stamp_ns/1e9, pixels, K, role=role)
            return record_attempt(args.output, report, role, acquire, tx)

        dump(args.output/'manifest.json', report)
        if args.once:
            save_capture('fit')
        else:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL);cv2.resizeWindow(title, 1100, 619)
            message = 'Read only. Keep camera fixed after framing; operator positions arm.'
            last_preview = 0.
            while True:
                native, _ = image(); upright = native[::-1, ::-1].copy()
                corners, ids, _ = detector.detectMarkers(native)
                selected = marker_pixels(corners, ids)
                visible = selected is not None
                if visible:
                    pts = selected
                    pts = np.c_[639-pts[:, 0], 359-pts[:, 1]].astype(np.int32)
                    cv2.polylines(upright, [pts], True, (0, 220, 0), 2)
                text = f'ID0 {"VISIBLE" if visible else "NOT VISIBLE"} | fit {report["fit_samples"]}/15 | pose check {report["pose_check_samples"]}/5'
                cv2.putText(upright, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .46, (0, 220, 0) if visible else (0, 0, 255), 1)
                cv2.putText(upright, message[:90], (8, 345), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
                if time.monotonic()-last_preview > 1:
                    temporary = args.output/'preview_temp.png'
                    if cv2.imwrite(str(temporary), upright):temporary.replace(args.output/'latest_upright.png')
                    last_preview = time.monotonic()
                cv2.imshow(title, upright);key = cv2.waitKey(10)&0xff
                if key in (ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:break
                if key in (ord('s'), ord('h')):message = save_capture('fit' if key == ord('s') else 'pose_check')
    except BaseException as exc:
        report['fatal_error'] = repr(exc)
        raise
    finally:
        finalize_session(args.output, report, cam, robot, cv2.destroyAllWindows if not args.once else None, tx)


if __name__ == '__main__':
    main()
