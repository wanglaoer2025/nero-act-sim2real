"""Unchanged original qualification harness with a new output root and sealed loader."""
import importlib.util
import json
from pathlib import Path

from arm_tuning import ROOT, INCREMENT
from arm_projection import load_projected
from pipeline import snapshot, require_unchanged


def main():
    specification = importlib.util.spec_from_file_location('original_increment_qualification', INCREMENT/'run_qualification.py')
    evaluator = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(evaluator)
    evaluator.ROOT = ROOT
    evaluator.load_policy = load_projected
    for directory in ['results/mechanism_original', 'results/nominal_after']:
        (ROOT/directory).mkdir(exist_ok=True)
    files = [Path(__file__), INCREMENT/'run_qualification.py', ROOT/'arm_projection.py',
             ROOT/'training/projected_A/policy_last.ckpt', ROOT/'training/projected_A/integrity.json']
    hashes = snapshot(files)
    evaluator.main()
    require_unchanged(hashes)
    binding = dict(policy_profile='projected_increment_arm_rows_v1',
                   actual_checkpoint=str(ROOT/'training/projected_A/policy_last.ckpt'),
                   actual_optimizer_updates=1000, additional_optimizer_updates=0,
                   unchanged_evaluator=str(INCREMENT/'run_qualification.py'), source_sha256=hashes,
                   hardware_commands_sent=False)
    (ROOT/'results/qualification/projected_policy_binding.json').write_text(json.dumps(binding, indent=2)+'\n')


if __name__ == '__main__':
    main()
