"""Both candidates use identical additional rejection; all targets remain ACT outputs."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'nero_act_full_flow_repair_20261023'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('sealed_full_flow_evaluation', SOURCE/'evaluate_repair.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
from contact_guard import PregraspSweptGuard
from executor.release_hold import hold_open_before_grasp_depth
sys.path.insert(0, str(ROOT.parent/'nero_act_sampling_retention_20261023'))
from predictive_guard import PredictivePhaseGuard
import simulation.supervised_rollout as runtime


def main():
    evaluation.ROOT = ROOT
    original_contract, original_contact, original_phases = runtime.FullTaskContract, runtime.PhysicsContact, runtime.PhaseGuard
    original_geometry = runtime.RobotGeometry
    contact_context = {}
    counts = dict(prefixes_checked=0, interpolated_states_checked=0, rejected_prefixes=0)

    class SceneGeometry(original_geometry):
        def __init__(self, scene):
            super().__init__(scene)
            contact_context['swept'] = PregraspSweptGuard(scene)
            contact_context['geometry'] = self
            contact_context['grasp_height'] = scene['table_top_z_m']+scene['grasp_tcp_height_m']

    class ObservedPhysicsContact(original_contact):
        def __init__(self):
            super().__init__()
            contact_context['contact'] = self

    class CollisionContract(original_contract):
        def validate_prefix(self, raw, measured, previous):
            rows = super().validate_prefix(raw, measured, previous)
            contact = contact_context['contact']
            inspected = rows[:2]
            current_tcp = contact_context['geometry'].pose(measured['state'])['tcp']
            if (not contact.close_authorized
                    and abs(current_tcp[2]-contact_context['grasp_height'])>.002):
                inspected = hold_open_before_grasp_depth(inspected, previous)
            try:
                checked = contact_context['swept'].validate(measured['state'], inspected, close_authorized=contact.close_authorized)
            except RuntimeError:
                counts['rejected_prefixes'] += 1
                raise
            counts['prefixes_checked'] += 1
            counts['interpolated_states_checked'] += checked
            return rows

    try:
        runtime.FullTaskContract = CollisionContract
        runtime.PhysicsContact = ObservedPhysicsContact
        runtime.PhaseGuard = PredictivePhaseGuard
        runtime.RobotGeometry = SceneGeometry
        evaluation.main()
    finally:
        runtime.FullTaskContract, runtime.PhysicsContact, runtime.PhaseGuard = original_contract, original_contact, original_phases
        runtime.RobotGeometry = original_geometry
        (ROOT/'collision_check_audit.json').write_text(json.dumps(dict(counts,
            simulation_only=True, motion_targets_generated_or_rewritten=False,
            source_sha256=evaluation.snapshot([Path(__file__), ROOT/'contact_guard.py'])), indent=2)+'\n')


if __name__ == '__main__':
    main()
