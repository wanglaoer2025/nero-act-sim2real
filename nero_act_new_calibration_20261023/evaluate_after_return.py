"""Offline saved-frame ACT inference and full-task tests using the new hand-eye fit.

No camera, ROS or SDK connection. A hypothetical 95mm preparation is separately
labelled and cannot be interpreted as an action performed on the robot.
"""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
POLICY_ROOT = ROOT.parent/'nero_act_preservation_scope_20261023'
OUTPUT = ROOT/'after_return_validation'


def build_cases(scene):
    measured = copy.deepcopy(scene)
    prepared = copy.deepcopy(scene)
    prepared['initial_robot_state']['gripper_total_width_m'] = .095
    return [dict(name='measured_start', scene=measured, hypothetical_preparation=False),
            dict(name='hypothetical_95mm_start', scene=prepared, hypothetical_preparation=True)]


def main():
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT.parent/'nero_act_dense_response_20261023'))
    import evaluate_dense
    from executor.contract import FullTaskContract
    from executor.model import make_policy_observation, physical_actions
    from executor.ros_execution import RobotGeometry
    from src.act.nero_dataset import preprocess_images

    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'contact_guard.py').symlink_to(ROOT/'contact_guard.py')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)

    def evaluate():
        evaluation = evaluate_dense.evaluation
        evaluation.ROOT = POLICY_ROOT
        policy, stats = evaluation.load_candidate()
        scene = json.loads((ROOT/'scene_relocalized_after_return.json').read_text())
        source = ROOT/'scene_capture_after_return'
        protected = evaluation.snapshot([Path(__file__), ROOT/'scene_relocalized_after_return.json',
            ROOT/'inputs/candidate_camera_to_base.json', source/'report.json', source/'feedback.json',
            source/'workspace_sensor_localization.json', source/'upright.png',
            POLICY_ROOT/'training/pilot_A/policy_last.ckpt'])
        state = np.r_[scene['initial_robot_state']['qpos_rad'],
                      scene['initial_robot_state']['gripper_total_width_m']].astype(np.float32)
        sensor = json.loads((source/'workspace_sensor_localization.json').read_text())['localization']
        geometry = RobotGeometry(scene)
        tcp = geometry.pose(state)['tcp']
        features = np.r_[np.asarray(sensor['source_bottom_mm'])/1000-tcp,
                         np.asarray(sensor['destination_mm'])/1000-tcp]
        issue, input_issue, raw = None, None, None
        try:
            observation = make_policy_observation(state, stats, policy.config, float(state[7]),
                                                   geometry_features_m=features)
        except ValueError as exc:
            input_issue = str(exc)
        if input_issue is None:
            rgb = cv2.cvtColor(cv2.imread(str(source/'upright.png')), cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                raw = physical_actions(policy(torch.from_numpy(observation)[None].cuda(),
                    preprocess_images(rgb[None], policy.config['image_size'])[None].cuda()),
                    stats, bounded_gripper=True)[0].cpu().numpy()
            assert raw.shape == (20, 8) and np.isfinite(raw).all()
            try:
                # Zero age isolates action/history structure ONLY; this saved frame is not live.
                FullTaskContract(policy.config).validate_prefix(raw, dict(state=state, age_s=0), state)
            except ValueError as exc:
                issue = str(exc)
        prediction = dict(mode='saved_frame_offline_ACT_diagnostic', live_shadow=False,
            hardware_commands_sent=False, freshness_qualified=False,
            structural_check_assumes_zero_age=True, structural_contract_error=issue,
            input_encoding_error=input_issue, ACT_prediction_generated=raw is not None,
            actual_state=state.tolist(), tcp_m=tcp.tolist(), geometry_features_m=features.tolist(),
            raw_first_two=None if raw is None else raw[:2].tolist(),
            first_two_delta_from_feedback=None if raw is None else (raw[:2]-state).tolist(),
            previous_gripper_target_is_hypothetical_measured_hold=True,
            real_submitted_reference_history_available=False)
        (OUTPUT/'saved_frame_prediction.json').write_text(json.dumps(prediction, indent=2)+'\n')
        print(json.dumps(dict(stage='saved_frame', input_encoding_error=input_issue,
                              structural_contract_error=issue)), flush=True)
        reports = []
        for case in build_cases(scene):
            scenario = dict(name=case['name'], seed=191899, query_stride=2,
                servo_delay_steps=0, observation_age_s=.05, jitter=False, force=False)
            torch.manual_seed(42)
            np.random.seed(42)
            result = evaluate_dense.runtime.rollout(case['scene'], policy, stats, scenario,
                OUTPUT/case['name'], runtime_profile='history_common_v1')
            record = dict(name=case['name'], hypothetical_preparation=case['hypothetical_preparation'],
                preparation_performed_on_robot=False, full_task_passed=evaluation.full_task_passed(result),
                initial_state=case['scene']['initial_robot_state'], result=result)
            reports.append(record)
            print(json.dumps(dict(stage=case['name'], success=result['success'], error=result['error'],
                active_policy_points=result['active_policy_points'], max_lift_m=result['max_lift_m'])), flush=True)
        evaluation.require_unchanged(protected)
        summary = dict(complete=True, results=reports, optimizer_updates=0,
            source_sha256=protected, actual_objects_relocalized=True, actual_pose_read=True,
            live_shadow=False, real_contact_geometry_qualified=False,
            hardware_commands_sent=False, hardware_deployment_allowed=False)
        (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    evaluate_dense.ROOT = OUTPUT
    evaluate_dense.evaluation.main = evaluate
    evaluate_dense.main()


if __name__ == '__main__':
    main()
