"""Execute the sealed seven-case full-task regression on CPU; no training or hardware."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SOURCE = WORKSPACE / 'nero_act_contact_escape_fix_20261023'
loader_root = WORKSPACE / 'nero_act_real_visual_bridge_20261024'
sys.path.insert(0, str(loader_root))
loader_spec = importlib.util.spec_from_file_location('_acceptance_cpu_loader', loader_root/'experiment.py')
loader = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(loader)
load_baseline = loader.load_baseline

sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('_acceptance_sealed_evaluator', SOURCE / 'evaluate_escape.py')
sealed = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sealed
spec.loader.exec_module(sealed)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    torch.set_num_threads(2)
    output = ROOT / 'run_01'
    output.mkdir(exist_ok=False)
    source_protocol = SOURCE / 'evaluation_protocol.json'
    original = json.loads(source_protocol.read_text())
    rows = original['cases']
    assert [r['id'] for r in rows] == [9000, 9005, 9006, 9200, 9201, 9300, 9301]
    protocol = dict(
        checkpoint_sha256=original['checkpoint_sha256'], cases=rows,
        expected_case_ids=[r['id'] for r in rows], scope='seven_existing_development_regressions',
        fresh_independent_position_holdout=False, visual_adapter_attached=False,
        compute_device='cpu', renderer='Mesa EGL llvmpipe',
        compute_and_render_device_changed_from_historical_gpu=True,
        runtime_profile='history_common_v1', simulated_confirmation_only=True,
        hypothetical_95mm_preparation=True, real_preparation_performed=False,
        additional_training_updates=0, hardware_commands_sent=False,
        retained_action_holds=sealed.action_provenance(),
        criteria=dict(minimum_lift_m=.18, minimum_transfer_exclusive_m=.1,
            maximum_target_error_exclusive_m=.02, minimum_stable_released_frames=100,
            maximum_active_policy_points=3900, abnormal_contacts_allowed=0,
            contract_failures_allowed=0, premature_release_allowed=False,
            required_events=['GRASP', 'CARRY', 'RELEASE', 'RETREAT']),
        rendering_environment={key:os.environ.get(key) for key in
            ['MUJOCO_GL','PYOPENGL_PLATFORM','EGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE','__EGL_VENDOR_LIBRARY_FILENAMES']})
    write(output / 'protocol.json', protocol)
    policy, stats = load_baseline()
    frozen = {k:v.clone() for k,v in policy.state_dict().items()}
    assert not any(p.requires_grad for p in policy.parameters())
    dependencies = sealed.actual_sources(sealed)
    simulation_root = Path(sys.modules['simulation'].__file__).resolve().parent
    dependencies += [p for p in simulation_root.rglob('*') if p.is_file() and
        p.suffix.lower() in {'.py','.xml','.stl','.obj','.msh','.urdf','.json','.png'}]
    dependencies += [*SOURCE.glob('*.py'), Path(__file__), source_protocol,
        output/'protocol.json', WORKSPACE/'nero_act_real_visual_bridge_20261024/experiment.py',
        WORKSPACE/'nero_act_real_visual_bridge_20261024/bridge.py',
        WORKSPACE/'nero_act_new_handeye_policy_repair_20261023/training/pilot_A/policy_last.ckpt']
    hashes = sealed.historical.snapshot(dependencies)
    write(output / 'source_sha256.json', hashes)
    records = []
    summary = dict(complete=False, finalized=False, full_suite_passed=False, records=records,
        model='1a28 frozen original', compute_device='cpu', renderer='Mesa EGL llvmpipe',
        all_cases_existing_development_data=True, independent_generalization_accepted=False,
        real_hardware_tested=False, hardware_commands_sent=False, hardware_deployment_allowed=False,
        visual_adapter_adopted=False, additional_training_updates=0)
    write(output / 'summary.json', summary)
    sealed.visual_evaluation.NominalImageEstimator = sealed.CardEdgesEstimator
    sealed.evaluate_dense.runtime = sealed.runtime
    sealed.evaluate_dense.ROOT = ROOT

    def evaluate():
        for row in rows:
            torch.manual_seed(42)
            np.random.seed(42)
            count = 0
            started = time.monotonic()
            def heartbeat(_module, _inputs, _output):
                nonlocal count
                count += 1
                if count % 100 == 0:
                    progress = dict(case_id=row['id'],queries=count,elapsed_s=time.monotonic()-started)
                    write(output/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            hook = policy.register_forward_hook(heartbeat)
            scenario = dict(name=f'cpu_full_acceptance_{row["id"]}',seed=191899,
                query_stride=2,servo_delay_steps=0,observation_age_s=.05,jitter=False,force=False)
            print(json.dumps(dict(start_case=row['id'],offset_xy_m=row['offset_xy_m'])),flush=True)
            try:
                result = sealed.runtime.rollout(row['scene'],policy,stats,scenario,
                    output/str(row['id']),runtime_profile='history_common_v1')
            finally:
                hook.remove()
            passed = sealed.evaluate_dense.evaluation.full_task_passed(result)
            records.append(dict(id=row['id'],offset_xy_m=row['offset_xy_m'],
                role='existing_development_regression',full_task_passed=passed,result=result))
            write(output/'summary.json',summary)
            print(json.dumps(dict(case_id=row['id'],passed=passed,active_points=result['active_policy_points'],
                lift_m=result['max_lift_m'],transfer_m=result['maximum_horizontal_transfer_m'],
                stable_released_frames=result['stable_released_frames'],
                events=[e['kind'] for e in result['simulated_events']],error=result['error'])),flush=True)

    sealed.evaluate_dense.evaluation.main = evaluate
    sealed.evaluate_dense.main()
    sealed.historical.require_unchanged(hashes)
    changed = [k for k,v in policy.state_dict().items() if not torch.equal(v,frozen[k])]
    assert not changed
    assert [r['id'] for r in records] == protocol['expected_case_ids']
    summary.update(complete=True,finalized=True,total=len(records),passed=sum(r['full_task_passed'] for r in records),
        full_suite_passed=all(r['full_task_passed'] for r in records),original_model_changed_tensors=changed,
        all_recorded_sources_unchanged=True,protected_source_count=len(hashes))
    write(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)


if __name__ == '__main__':
    main()
