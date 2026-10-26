import importlib
import importlib.util
import json
import signal
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def api():
    assert importlib.util.find_spec('capture') is not None, 'Passive calibration sampler missing'
    return importlib.import_module('capture')


def inputs():
    K = np.array([[600., 0, 320], [0, 600., 180], [0, 0, 1]])
    h = .095/2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    rv = Rotation.from_euler('xyz', [2.6, .2, .1]).as_rotvec()
    pixels = cv2.projectPoints(obj, rv, np.array([.01, -.02, .6]), K, None)[0].reshape(4, 2)
    def health(stamp):
        return dict(monotonic=stamp, timestamp=stamp, feedback_timestamps=[stamp]*19,
            feedback=dict(arm_status=2, err_code=0, ctrl_mode=1, motion_status=0),
            state=[0.]*7+[.095], flange_pose=[.1, .2, .3, 0., 0., 0.])
    window = [health(t) for t in [99.4, 99.7, 100.]]
    return K, pixels, window, health(100.02)


def test_measured95mm_pose_and_raw_status_preserved():
    m = api(); K, pixels, window, after = inputs()
    record = m.make_sample(window, after, 100.01, pixels, K, role='fit')
    np.testing.assert_allclose(np.array(record['transform_camera_from_marker_m'])[:3, 3],
                               [.01, -.02, .6], atol=1e-7)
    np.testing.assert_allclose(np.array(record['transform_base_from_flange_m'])[:3, 3], [.1, .2, .3])
    assert record['marker_black_side_m'] == .095
    assert record['health_after']['feedback']['arm_status'] == 2
    assert record['independent_absolute_validation'] is False
    assert record['calibration_verified'] is False and record['motion_sent'] is False


@pytest.mark.parametrize('problem', ['short_window', 'moving', 'camera_skew', 'nonzero_error',
                                      'wrong_mode', 'unknown_status'])
def test_unstable_or_unhealthy_pairs_rejected(problem):
    m = api(); K, pixels, window, after = inputs()
    stamp = 100.01
    if problem == 'short_window': window = window[1:]
    elif problem == 'moving': after['flange_pose'][0] += .001
    elif problem == 'camera_skew': stamp = 100.1
    elif problem == 'nonzero_error': after['feedback']['err_code'] = 16
    elif problem == 'wrong_mode': after['feedback']['ctrl_mode'] = 6
    else: after['feedback']['arm_status'] = 7
    with pytest.raises(ValueError): m.make_sample(window, after, stamp, pixels, K, role='fit')


def test_kept_out_pose_is_not_independent_metrology():
    m = api(); K, pixels, window, after = inputs()
    record = m.make_sample(window, after, 100.01, pixels, K, role='pose_check')
    assert record['role'] == 'pose_check'
    assert record['independent_absolute_validation'] is False


def test_missing_or_multiple_markers_cannot_save_a_pose():
    m = api(); K, pixels, window, after = inputs()
    for invalid in [None, np.zeros((3, 2)), np.full((4, 2), np.nan)]:
        with pytest.raises(ValueError): m.make_sample(window, after, 100.01, invalid, K, role='fit')


def test_sdk_fixed_xyz_rotation_convention():
    T = api().flange_transform([.1, .2, .3, np.pi/2, 0, 0])
    np.testing.assert_allclose(T[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]], atol=1e-12)


def test_nearly_frontal_ambiguous_square_cannot_become_a_fit_sample():
    m = api(); K, _, window, after = inputs()
    h = .095/2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    rv = np.array([np.pi-.03, 0., 0.])
    pixels = cv2.projectPoints(obj, rv, np.array([0., 0., .6]), K, None)[0].reshape(4, 2)
    with pytest.raises(ValueError, match='ambiguous'):
        m.make_sample(window, after, 100.01, pixels, K, role='fit')


def test_help_opens_neither_camera_nor_robot():
    module = api()
    result = subprocess.run([sys.executable, str(Path(module.__file__)), '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0 and '--output' in result.stdout


@pytest.mark.parametrize('role', ['fit', 'pose_check'])
def test_same_flange_pose_does_not_count_again_across_roles(role):
    m = api(); K, pixels, window, after = inputs()
    sample = m.make_sample(window, after, 100.01, pixels, K, role=role)
    prior = dict(accepted=True, role='fit', transform_base_from_flange_m=
                 sample['transform_base_from_flange_m'])
    with pytest.raises(ValueError, match='Duplicate'):
        m.require_distinct_pose(sample, [prior])
    shifted = np.array(prior['transform_base_from_flange_m'])
    shifted[0, 3] += .010
    prior['transform_base_from_flange_m'] = shifted.tolist()
    m.require_distinct_pose(sample, [prior])


def test_camera_identity_and_optical_rotation_are_unambiguous():
    m = api(); K, _, _, _ = inputs()
    profile = {'camera_serial': '39602891'}
    with pytest.raises(ValueError, match='serial'):
        m.camera_metadata(profile, 11111111, K)
    metadata = m.camera_metadata(profile, 39602891, K)
    assert metadata['image_rotation_degrees'] == 0
    assert metadata['preview_rotation_degrees'] == 180
    assert metadata['pixel_space'] == 'rectified_left'


def test_failed_json_replacement_preserves_previous_manifest(tmp_path, monkeypatch):
    m = api(); path = tmp_path/'manifest.json'
    m.dump(path, {'events': ['previous']})
    def fail_replace(*_):
        raise OSError('injected replace failure')
    monkeypatch.setattr(m.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='injected'):
        m.dump(path, {'events': ['new']})
    assert json.loads(path.read_text()) == {'events': ['previous']}


def test_only_single_requested_marker_is_selected():
    m = api(); _, pixels, _, _ = inputs()
    assert m.marker_pixels([pixels], np.array([[1]])) is None
    assert m.marker_pixels([pixels, pixels], np.array([[0], [0]])) is None
    np.testing.assert_array_equal(m.marker_pixels([pixels, pixels+20], np.array([[0], [1]])), pixels)
    np.testing.assert_array_equal(m.marker_pixels([pixels+20, pixels], np.array([[17], [0]])), pixels)
    np.testing.assert_array_equal(m.marker_pixels([pixels], np.array([[0]])), pixels)


def test_image_write_failure_leaves_rejected_attempt_on_disk(tmp_path, monkeypatch):
    m = api(); K, pixels, window, after = inputs()
    report = dict(events=[], camera_metadata={})
    def acquire(payload):
        payload['native'] = np.zeros((360, 640, 3), np.uint8)
        payload['sample'] = m.make_sample(window, after, 100.01, pixels, K, role='fit')
    monkeypatch.setattr(m.cv2, 'imwrite', lambda *_: False)
    m.record_attempt(tmp_path, report, 'fit', acquire, lambda: 907)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    attempt = json.loads((tmp_path/'attempt_000/sample.json').read_text())
    assert manifest['fit_samples'] == 0
    assert attempt['event']['accepted'] is False
    assert 'save' in attempt['event']['error']


def test_interrupted_capture_is_recorded_and_propagated(tmp_path):
    m = api(); report = dict(events=[], camera_metadata={})
    def acquire(_):
        raise KeyboardInterrupt('operator closed capture')
    with pytest.raises(KeyboardInterrupt):
        m.record_attempt(tmp_path, report, 'fit', acquire, lambda: 907)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['events'][0]['state'] == 'interrupted'
    assert manifest['fit_samples'] == 0
    assert (tmp_path/'attempt_000/sample.json').exists()


def test_duplicate_pose_cannot_increment_manifest_counts(tmp_path):
    m = api(); K, pixels, window, after = inputs()
    report = dict(events=[], camera_metadata={})
    def acquire(payload):
        payload['native'] = np.zeros((360, 640, 3), np.uint8)
        payload['sample'] = m.make_sample(window, after, 100.01, pixels, K, role=payload['event']['role'])
    m.record_attempt(tmp_path, report, 'fit', acquire, lambda: 907)
    m.record_attempt(tmp_path, report, 'pose_check', acquire, lambda: 907)
    manifest = json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['fit_samples'] == 1 and manifest['pose_check_samples'] == 0
    assert manifest['events'][1]['accepted'] is False
    assert 'Duplicate' in manifest['events'][1]['error']


def test_termination_signal_reaches_python_cleanup():
    api()
    code = ('import signal, capture\n'
            'capture.install_cleanup_signals()\n'
            'try:\n'
            ' print("ready", flush=True)\n'
            ' signal.pause()\n'
            'except KeyboardInterrupt:\n'
            ' print("cleanup", flush=True)\n')
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'ready'
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=5)
        assert process.returncode == 0 and out.strip() == 'cleanup', err
    finally:
        if process.poll() is None:
            process.kill(); process.wait()


def test_cleanup_failure_does_not_skip_other_cleanup_or_final_record(tmp_path):
    m = api(); calls = []
    class Camera:
        def close(self):
            calls.append('camera')
            raise RuntimeError('camera close failed')
    class Robot:
        def disconnect(self): calls.append('robot')
    def gui_close(): calls.append('gui')
    def missing_tx(): raise OSError('CAN no longer present')
    report = dict(running=True)
    m.finalize_session(tmp_path, report, Camera(), Robot(), gui_close, missing_tx)
    saved = json.loads((tmp_path/'manifest.json').read_text())
    assert calls == ['camera', 'robot', 'gui']
    assert saved['running'] is False
    assert len(saved['cleanup_errors']) == 2
    assert saved['can_tx_end'] is None
