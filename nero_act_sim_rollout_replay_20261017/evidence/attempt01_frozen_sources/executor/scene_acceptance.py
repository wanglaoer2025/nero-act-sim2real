"""A current user-approved cup-scene adjustment, never an action or limit change."""
import copy,hashlib,json
from pathlib import Path
import numpy as np

def scene_hash(scene):
    return hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest()

def accepted_scene(scene,manifest,path):
    result=copy.deepcopy(scene);path=Path(path)
    if not path.exists():return result
    acceptance=json.loads(path.read_text());point=np.asarray(acceptance.get('source_xy_m'),float)
    if (acceptance.get('user_statement')!='有一点偏差也没关系'
            or acceptance.get('trace_sha256')!=manifest.get('source_rollout_sha256')
            or acceptance.get('authorized_scene_sha256')!=scene_hash(scene)):
        raise PermissionError('Current user acceptance must bind original scene and unchanged ACT trace')
    limit=acceptance.get('maximum_accepted_source_offset_m')
    if (point.shape!=(2,) or not np.isfinite(point).all() or limit!=.01
            or np.linalg.norm(point-np.asarray(scene['pick_xy_m']))>limit):
        raise ValueError('Accepted source adjustment exceeds this trial10mm scope')
    result['pick_xy_m']=point.tolist()
    return result

def execution_scene(root,scene,manifest,*,require_evidence=False,receipt=None):
    root=Path(root);path=root/'evidence/source_offset_acceptance.json'
    result=accepted_scene(scene,manifest,path)
    if not path.exists():return result
    acceptance=json.loads(path.read_text())
    if receipt is not None:
        canonical=json.dumps(receipt,sort_keys=True,separators=(',',':'),allow_nan=False)
        if acceptance.get('parent_authorization_sha256')!=hashlib.sha256(canonical.encode()).hexdigest():
            raise PermissionError('Scene addendum does not match the current ACT_REPLAY authorization')
    if require_evidence:
        for name in ['accepted_offset_replay_physics.json','accepted_offset_collision/report.json']:
            proof=json.loads((root/'evidence'/name).read_text())
            if not (proof.get('success') is True and proof.get('hardware_connected') is False
                    and proof.get('execution_scene_sha256')==scene_hash(result)
                    and proof.get('trace_sha256')==manifest['source_rollout_sha256']):
                raise PermissionError('Adjusted cup scene lacks matching physical simulation/collision proof')
    return result
