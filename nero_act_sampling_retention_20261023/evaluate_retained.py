"""Evaluate the sampling repair with predictive-only safety checks in both arms of the comparison."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'nero_act_full_flow_repair_20261023'
sys.path.insert(0,str(PREVIOUS))
spec=importlib.util.spec_from_file_location('sealed_full_flow_assessment',PREVIOUS/'evaluate_repair.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
from predictive_guard import PredictivePhaseGuard
import simulation.supervised_rollout as physical_runtime


def main():
    evaluation.ROOT=ROOT
    original=physical_runtime.PhaseGuard
    physical_runtime.PhaseGuard=PredictivePhaseGuard
    try:evaluation.main()
    finally:physical_runtime.PhaseGuard=original
    report=ROOT/'results/full_flow_summary.json'
    result=json.loads(report.read_text())
    result.update(predictive_guard_enabled=True,motion_targets_generated_by_guard=False,
        predictive_guard_sha256=hashlib.sha256((ROOT/'predictive_guard.py').read_bytes()).hexdigest(),
        comparison_uses_same_stricter_guard=True)
    report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
