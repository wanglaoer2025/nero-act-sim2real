"""Pure frozen-evaluation bindings; imports neither ROS nor the policy namespace."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPERVISED_PROTOCOL='physics_pauses_first_close_support_hold_v6'
ARTIFACTS = [('policy.ckpt', 'checkpoint_sha256'), ('dataset_stats.pkl', 'stats_sha256'),
             ('scene_config.json', 'scene_sha256'), ('training_config.json', 'training_config_sha256')]


def evaluation_sources():
    names = ['executor/scene_acceptance.py','evidence/source_offset_acceptance.json','run.py','executor/replay.py','executor/replay_session.py','executor/replay_trace.py','executor/camera.py','replay/manifest.json','replay/act_rollout.npz','replay/simulation_result.json','executor/supervision.py','executor/transport.py','executor/permission.py','executor/policy_worker.py','executor/shadow.py','executor/driver.py','executor/gate.py','executor/execution_guards.py','executor/model.py', 'executor/contract.py', 'executor/bundle.py',
        'executor/evaluation_integrity.py', 'simulation/evaluate.py',
        'simulation/supervised_rollout.py','executor/release_hold.py','executor/phases.py',
        'executor/runtime.py','executor/ros_execution.py','executor/contact.py','executor/gate_health.py',
        'simulation/scene.py', 'simulation/teacher.py', 'simulation/outcome.py', 'simulation/snapshot.py',
        'simulation/nero_domain_randomization.py', 'simulation/nero_scene_camera.py',
        'simulation/scene_template.xml', 'scene/config.json']
    names += ['executor/approach_clearance.py','executor/current_replay_run.py']
    return sorted([ROOT/name for name in names]+list((ROOT/'policy_source').rglob('*.py'))+
                  [p for p in (ROOT/'simulation/assets').rglob('*') if p.is_file()])


def verify_frozen_artifacts(checkpoint, protocol, *, source_root=ROOT):
    directory, source_root = Path(checkpoint).parent, Path(source_root).resolve()
    for name, key in ARTIFACTS:
        if (not protocol.get(key)
                or hashlib.sha256((directory/name).read_bytes()).hexdigest() != protocol[key]):
            raise RuntimeError('Frozen evaluation artifact changed or unbound: '+name)
    if not protocol.get('source_sha256'):
        raise RuntimeError('Frozen evaluation has no source bindings')
    for name, expected in protocol['source_sha256'].items():
        path = (source_root/name).resolve()
        if not path.is_relative_to(source_root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Frozen evaluation source changed: '+name)
