"""One exact paired offline case to diagnose the new approach stall, no model updates."""
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from executor.policy_worker import FrozenAct
from simulation.supervised_rollout import rollout


def main():
    source=ROOT.parent/'nero_act_gripper_contact_focus_20261018/model/run26_current_fixed_scene'
    scene=json.loads((source/'scene_config.json').read_text())
    assert scene==json.loads((ROOT/'model/run27_fixed_heads/scene_config.json').read_text())
    current=ROOT/'evaluation/run27_current_scene14'
    protocol=json.loads((current/'protocol.json').read_text())
    case=next(c for c in protocol['scenarios'] if c['name']=='independent_0')
    output=ROOT/'evidence/paired_previous_policy';output.mkdir(exist_ok=False)
    old=FrozenAct(source,device='cuda')
    old.model.config=old.config;old.model.device=torch.device('cuda')
    result=rollout(scene,old.model,old.stats,case,output/'previous_same_case')
    newer=json.loads((current/'independent_0/result.json').read_text())
    report=dict(mode='single_paired_development_diagnostic',same_scene_and_scenario=True,
        scenario=case,previous_checkpoint_sha256=old.metadata['checkpoint_sha256'],
        previous_success=result['success'],current_success=newer['success'],
        previous_events=result['simulated_events'],current_events=newer['simulated_events'],
        previous_active_points=result['active_policy_points'],current_active_points=newer['active_policy_points'],
        previous_last_tcp_z_mm=result['last_observation']['tcp_xyz'][2]*1000,
        current_last_tcp_z_mm=newer['last_observation']['tcp_xyz'][2]*1000,
        eligibility_for_hardware=False,hardware_connected=False)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
