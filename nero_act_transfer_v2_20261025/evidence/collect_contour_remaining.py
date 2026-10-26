"""One bounded continuation of the approved150-episode simulation collection."""
import os
os.environ['MUJOCO_GL'] = 'egl'
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, CancelledError
import h5py

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect_sim import collect_one, planned_jobs
from rebuild_data import write
from pilot_revalidate import sha


if __name__ == '__main__':
    output = ROOT / 'data/v2_contour_01'
    path = output / 'manifest.json'
    manifest = json.loads(path.read_text())
    assert len(manifest['episodes']) == 131 and not manifest['rejected']
    assert manifest['observer_sha256'] == sha(ROOT / 'observer.py')
    assert not manifest['collection_running'] and not manifest['complete']
    binding = json.loads((ROOT / 'evidence/implementation_manifest.json').read_text())
    assert all(sha(ROOT / name) == digest for name, digest in binding['sha256'].items())
    with h5py.File(manifest['source_scene']) as source:
        base = json.loads(source.attrs['scene_config'])
    protocol = json.loads((ROOT / 'position_protocol.json').read_text())
    jobs = planned_jobs(base, output, protocol)
    seen = {r['path'] for r in manifest['episodes']}
    assert len(seen) == 131 and seen.issubset({j['output'] for j in jobs})
    todo = [j for j in jobs if j['output'] not in seen]
    assert len(todo) == 19 and not any(Path(j['output']).exists() for j in todo)
    # Existing immutable131 were source-hashed in the bound equivalence audit;
    # train.preflight will rehash the full150 again before any optimization.
    manifest['collection_running'] = True
    write(path, manifest)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(collect_one, job) for job in todo]
        for future in as_completed(futures):
            try:
                row = future.result()
            except CancelledError:
                continue
            except Exception as exc:
                row = dict(accepted=False, error=type(exc).__name__ + ': ' + str(exc))
            manifest['episodes' if row['accepted'] else 'rejected'].append(row)
            if not row['accepted']:
                manifest['stopped_on_failure'] = True
                for pending in futures:
                    pending.cancel()
            write(path, manifest)
            print(json.dumps(dict(accepted=len(manifest['episodes']), rejected=len(manifest['rejected']),
                last=row.get('position_group'), error=row.get('error'))), flush=True)
    manifest['collection_running'] = False
    manifest['complete'] = len(manifest['episodes']) == 150 and not manifest['rejected']
    write(path, manifest)
    raise SystemExit(0 if manifest['complete'] else 2)
