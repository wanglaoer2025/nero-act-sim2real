"""Catch mismatched recovery starts and unsafe teacher labels with actual physics."""
import json
from pathlib import Path

import numpy as np

from lift_recovery import collect_lift_prefix


ROOT = Path(__file__).resolve().parents[1]


def test_exact_failed_state_teacher_lifts_without_weakening_attachment_guard(tmp_path):
    source = ROOT / 'results/exact_lift_failure'
    scene = json.loads((source / 'scene_configuration.json').read_text())
    context = json.loads((source / 'runtime_checkpoints.json').read_text())['records'][-1]
    report = collect_lift_prefix(scene, context, tmp_path / 'lift.npz')
    assert report['verified']
    assert report['final_cup_lift_m'] >= .02
    assert report['minimum_predicted_cup_bottom_m'] >= -1e-6
    assert report['unexpected_contact_frames'] == 0
    assert 20 <= report['frames'] <= 400
    with np.load(tmp_path / 'lift.npz', allow_pickle=False) as episode:
        np.testing.assert_array_equal(episode['qpos'][0], context['physics']['qpos'])
        assert (episode['target_tcp_dz_m'][:2] > 0).all()
        assert np.max(abs(np.diff(episode['action'][:, :7], axis=0))) <= .0010001
        assert episode['holding'].all()
