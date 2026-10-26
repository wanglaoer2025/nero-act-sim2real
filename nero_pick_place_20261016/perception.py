"""Explicit YOLO-cup / unique HSV-blue localization; no robot connections."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNTIME = Path('/home/user/zyh/data/nero-teacher/runtime-v4')
CAMERA = Path('/home/user/wq/nero_cup_retrained_v2_20261016/scene/calibration/camera_to_base.json')
TCP = RUNTIME / 'config/tcp_offset.json'


def yellow_cup_box(rgb, cups):
    if len(cups) != 1:
        raise ValueError('Expected exactly one YOLO cup before yellow outline refinement')
    box = np.asarray(cups[0], float)
    height, width = rgb.shape[:2]
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not 0 < box[0] < box[2] < width-1 or not 0 < box[1] < box[3] < height-1):
        raise ValueError('YOLO cup box is invalid or truncated')
    x1, y1 = np.maximum(0, np.floor(box[:2]).astype(int)-8)
    x2, y2 = np.minimum([width, height], np.ceil(box[2:]).astype(int)+9)
    crop = rgb[y1:y2, x1:x2]
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV),
                       np.array([18, 70, 70]), np.array([40, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    if len(contours) != 1:
        raise ValueError('Expected exactly one substantial yellow cup contour inside YOLO region')
    points = contours[0][:, 0]
    lo, hi = points.min(0), points.max(0)
    if np.any(lo == 0) or np.any(hi >= np.array([x2-x1, y2-y1])-1):
        raise ValueError('Yellow contour touches crop border')
    refined = np.r_[lo+[x1, y1], hi+[x1, y1]]
    overlap = np.maximum(0, np.minimum(refined[2:], box[2:])-np.maximum(refined[:2], box[:2]))
    intersection = float(np.prod(overlap))
    union = float(np.prod(refined[2:]-refined[:2])+np.prod(box[2:]-box[:2])-intersection)
    if union <= 0 or intersection/union < .5:
        raise ValueError('Yellow outline disagrees with YOLO cup identity')
    return refined


def unique_blue_box(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([85, 80, 70]), np.array([125, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) <= 500:
            continue
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            raise ValueError('Large blue region is not a convex quadrilateral')
        points = polygon[:, 0]
        box = np.r_[points.min(0), points.max(0)]
        if not (0 < box[0] < box[2] < rgb.shape[1]-1
                and 0 < box[1] < box[3] < rgb.shape[0]-1):
            raise ValueError('Blue quadrilateral is truncated')
        candidates.append(box)
    if len(candidates) != 1:
        raise ValueError('Expected exactly one complete blue quadrilateral')
    return candidates[0]


def require_current_targets(data, now):
    age = now - data['capture_wall_s']
    if (not math.isfinite(age) or not 0 <= age <= 90
            or data.get('geometry_fit_passed') is not True
            or data.get('table_z_base_mm') != 0
            or data.get('tcp_approximate') is not True
            or data.get('source_method') not in ('YOLO cup', 'YOLO cup with unique yellow contour')
            or data.get('destination_method') != 'unique HSV blue quadrilateral'):
        raise ValueError('Fresh explicitly sourced historical-parameter geometry is required')


def revalidated_targets(original, observed, capture_wall_s):
    for key in ('source_bottom_mm', 'destination_mm'):
        point = np.asarray(observed[key], float)
        if (point.shape != (3,) or not np.isfinite(point).all()
                or np.linalg.norm(point-np.asarray(original['plan'][key])) > 5):
            raise ValueError('Target changed by more than the existing 5mm stability bound')
    if not math.isfinite(capture_wall_s):
        raise ValueError('Invalid camera capture time')
    return dict(original, capture_wall_s=capture_wall_s, revalidated_observation=observed)


def refresh_from_recorder(recorder, task_args, original, output, *, model):
    from scripts import nero_methylene_teacher as teacher
    cup_id = next(index for index, name in model.names.items() if name == 'cup')
    stability = teacher.TargetStability(8, 5)
    matrix = np.asarray(original['targets']['camera_matrix'])
    transform = np.asarray(original['targets']['base_from_camera'])
    deadline, previous = time.monotonic()+5, None
    while time.monotonic() < deadline:
        teacher.require_recording(recorder)
        rgb, stamp = recorder.camera.cameras['camera_0'].latest()
        if not 0 <= time.monotonic()-stamp <= .15:
            raise TimeoutError('Reacquisition camera feedback is stale')
        if stamp == previous:
            time.sleep(.005)
            continue
        previous = stamp
        capture_wall_s = time.time()-(time.monotonic()-stamp)
        if rgb.shape[:2] != (360, 640):
            raise ValueError('Reacquisition image differs from original intrinsics resolution')
        prediction = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), classes=[cup_id],
                                   conf=task_args.confidence, imgsz=960, verbose=False)[0]
        cups = [] if prediction.boxes is None else prediction.boxes.xyxy.cpu().numpy()
        try:
            refined_cups = ([yellow_cup_box(rgb, cups)]
                            if original['source_method'] == 'YOLO cup with unique yellow contour' else cups)
            observation = teacher.localize_pair(refined_cups, [unique_blue_box(rgb)], rgb.shape,
                                                matrix, transform, task_args)
            locked = stability.update(observation['source_bottom_mm'], observation['destination_mm'])
        except ValueError:
            stability.clear()
            continue
        if locked is not None:
            observation.update(source_bottom_mm=locked[0], destination_mm=locked[1])
            refreshed = revalidated_targets(original, observation, capture_wall_s)
            require_current_targets(refreshed, time.time())
            teacher.save_perception_diagnostic(output, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), observation)
            return refreshed
    raise TimeoutError('Could not freshly revalidate the same cup and blue card; stop')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(RUNTIME), '/home/user/zyh', '/home/user/zyh/src']
    from scripts import nero_methylene_teacher as teacher
    from src.utils.vision import validate_calibration_quality, validate_rigid_transform
    from ultralytics import YOLO
    observations = np.load(args.capture / 'observations.npz', allow_pickle=False)
    report = json.loads((args.capture / 'report.json').read_text())
    metadata = report['camera_metadata']
    intrinsics = metadata['rectified_left_intrinsics_native']
    scale = metadata['capture_width'] / metadata['native_width']
    matrix = np.array([[intrinsics['fx'], 0, intrinsics['cx']],
                       [0, intrinsics['fy'], intrinsics['cy']], [0, 0, 1.]])
    matrix[:2] *= scale
    calibration = json.loads(CAMERA.read_text())
    validate_calibration_quality(calibration)
    transform = validate_rigid_transform(calibration['transform_base_from_camera'])
    teacher_args = teacher.parse_args([
        '--weights', str(RUNTIME / 'models/yolo.pt'), '--pick-place',
        '--cup-height-mm', '95', '--cup-radius-mm', '37.5', '--table-z-base-mm', '0',
        '--act-output', str(args.capture / 'unused.hdf5'), '--act-position-group', '1',
        '--transit-speed-percent', '5', '--precision-speed-percent', '1',
        '--allow-approximate-tcp', '--confidence', '.2',
    ])
    model = YOLO(teacher_args.weights)
    cup_id = next(index for index, name in model.names.items() if name == 'cup')
    stability = teacher.TargetStability(8, 5)
    locked = None
    for rgb in observations['rgb']:
        prediction = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), classes=[cup_id],
                                   conf=.2, imgsz=960, verbose=False)[0]
        cups = [] if prediction.boxes is None else prediction.boxes.xyxy.cpu().numpy()
        targets = teacher.localize_pair([yellow_cup_box(rgb, cups)], [unique_blue_box(rgb)], rgb.shape,
                                        matrix, transform, teacher_args)
        locked = stability.update(targets['source_bottom_mm'], targets['destination_mm'])
    if locked is None:
        raise ValueError('Eight consecutive stable observations are required')
    plan = teacher.transfer_plan(*locked, teacher_args, json.loads(TCP.read_text()))
    output = dict(
        capture_wall_s=float(observations['sample_wall_unix_s'][-1]),
        geometry_fit_passed=True, table_z_base_mm=0, tcp_approximate=True,
        source_method='YOLO cup with unique yellow contour', destination_method='unique HSV blue quadrilateral',
        calibration_sha256=hashlib.sha256(CAMERA.read_bytes()).hexdigest(),
        tcp_sha256=hashlib.sha256(TCP.read_bytes()).hexdigest(),
        parameters=vars(teacher_args), targets=targets, plan=plan,
    )
    require_current_targets(output, time.time())
    with (args.capture / 'targets.json').open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2,
                  default=lambda value: str(value) if isinstance(value, Path) else np.asarray(value).tolist())
    image = cv2.cvtColor(observations['rgb'][-1], cv2.COLOR_RGB2BGR)
    teacher.save_perception_diagnostic(args.capture / 'locked', image, targets)
    print(json.dumps(dict(source=plan['source_bottom_mm'], destination=plan['destination_mm']),
                     default=lambda value: value.tolist()), flush=True)


if __name__ == '__main__':
    main()
