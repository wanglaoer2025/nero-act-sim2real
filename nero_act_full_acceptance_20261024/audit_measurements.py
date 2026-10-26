"""Cross-check completed case summaries against saved physics measurements only."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run_01'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    summary = json.loads((RUN/'acceptance.json').read_text())
    protocol = json.loads((RUN/'protocol.json').read_text())
    hashes = json.loads((RUN/'source_sha256.json').read_text())
    assert summary['complete'] and summary['finalized']
    assert [r['id'] for r in summary['records']] == protocol['expected_case_ids']
    for path, digest in hashes.items():
        assert sha256(path) == digest, path
    rows = []
    assets = {}
    for record in summary['records']:
        folder = RUN/str(record['id'])
        measures = json.loads((folder/'measurements.json').read_text())
        scene = json.loads((folder/'scene_configuration.json').read_text())
        result = json.loads((folder/'result.json').read_text())
        assert record['result'] == result and len(measures) == result['frames']
        cup = np.array([m['cup_xyz'] for m in measures])
        tcp = np.array([m['tcp_xyz'] for m in measures])
        bottom = cup[:,2] - scene['cup']['height_m']/2 - scene['table_top_z_m']
        target_error = np.linalg.norm(cup[:,:2]-scene['place_xy_m'],axis=1)
        transfer = np.linalg.norm(cup[:,:2]-scene['pick_xy_m'],axis=1)
        stable = np.array([bool(m['supported'] and m['cup_upright']>.95 and m['cup_speed']<.05
            and not m['bilateral'] and m['gripper_width']>.085) for m in measures])
        stable &= (target_error<.02) & (np.abs(bottom)<.002) & (tcp[:,2]>=scene['table_top_z_m']+.245)
        suffix = next((i for i, good in enumerate(stable[::-1]) if not good),len(stable))
        assert suffix == result['stable_released_frames']
        assert np.isclose(max(0.,float(bottom.max())),result['max_lift_m'],atol=1e-12,rtol=0)
        assert np.isclose(float(transfer.max()),result['maximum_horizontal_transfer_m'],atol=1e-12,rtol=0)
        assert sum(m['unexpected_contact'] for m in measures) == result['unexpected_arm_cup_contact_frames']
        if record['full_task_passed']:
            assert result['success'] and result['error'] is None and suffix>=100
            assert bottom.max()>=.18 and transfer.max()>.1
            assert [e['kind'] for e in result['simulated_events']]==['GRASP','CARRY','RELEASE','RETREAT']
            assert not result['premature_release_before_support']
            assert result['execution_contract_failures']==0
        rows.append(dict(id=record['id'],full_task_passed=record['full_task_passed'],
            max_lift_mm=float(max(0.,bottom.max())*1000),max_transfer_mm=float(transfer.max()*1000),
            final_target_error_mm=float(target_error[-1]*1000),stable_suffix_frames=int(suffix),
            final_cup_bottom_mm=float(bottom[-1]*1000),final_tcp_height_mm=float(tcp[-1,2]*1000),
            bilateral_contact_frames=sum(m['bilateral'] for m in measures),
            events=[e['kind'] for e in result['simulated_events']],error=result['error']))
        for path in folder.iterdir():
            if path.is_file():assets[str(path)]=sha256(path)
    report = dict(complete=True,measurement_count=sum(r['result']['frames'] for r in summary['records']),
        all_saved_summary_metrics_match_measurements=True,protected_sources_unchanged=len(hashes),
        total=7,passed=sum(r['full_task_passed'] for r in rows),cases=rows,
        independent_geometry_instrument=False,scope='recalculation_from_same_saved_simulated_physics',
        full_real_task_accepted=False,hardware_commands_sent=False,assets_sha256=assets)
    (RUN/'measurement_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='assets_sha256'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
