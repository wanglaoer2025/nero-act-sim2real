"""Fixed3000-update weights-only continuation of the unchanged ACT experiment."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from resume import ROOT,SOURCE,PROFILE,INITIALIZER,prior,load_initial
from objective import executed_objective,preservation_scale
from response import evaluate_response
from pipeline import snapshot,require_unchanged
from training.approach_preservation import ApproachPreservation,validate_approach_replay
from training.pose_objective import CartesianPoseLoss,fk_spec_from_scene


def main():
    torch.set_num_threads(2);torch.manual_seed(43);np.random.seed(43)
    plan=json.loads((ROOT/'plan.json').read_text())
    if plan['additional_updates']!=3000 or plan['batch_size']!=32:
        raise ValueError('This continuation is limited to3000 updates at batch32')
    output=ROOT/'training/pilot_A';output.mkdir(parents=True,exist_ok=False)
    policy,stats,groups=load_initial()
    dataset,old_batches,manifest,scene,files=prior.prepare_data(policy,stats)
    original=np.load(INITIALIZER.parent/'batch_order.npy',allow_pickle=False)
    if not np.array_equal(old_batches,original):raise ValueError('Prior training sampler changed')
    rng=np.random.default_rng(43)
    batches=np.concatenate([old_batches[rng.permutation(len(old_batches))] for _ in range(3)])
    np.save(output/'batch_order.npy',batches,allow_pickle=False)
    files+=list(ROOT.glob('*.py'))+[ROOT/'plan.json',ROOT/'data/scene_protocol.json',
        SOURCE/'plan.json',SOURCE/'data/pose_protocol.json',SOURCE/'data/scene_protocol.json',
        output/'batch_order.npy',INITIALIZER,INITIALIZER.parent/'integrity.json',
        INITIALIZER.parent/'batch_order.npy',INITIALIZER.parent/'response_gate.json',
        ROOT.parent/'nero_act_position_recovery_20261022/evaluate.py',
        ROOT.parent/'nero_act_position_adaptation_20261022/data/scene_protocol.json',
        ROOT.parent/'nero_act_command_increment_20261021/run_qualification.py']
    files += [Path(m.__file__) for m in tuple(sys.modules.values()) if getattr(m,'__file__',None)
        and str(m.__file__).startswith(str(ROOT.parent)+'/') and str(m.__file__).endswith('.py')]
    hashes=snapshot(files)
    hashes.update(validate_approach_replay(policy.config['approach_preservation']['replay_path']))
    keeper=ApproachPreservation(dict(policy.config['approach_preservation'],batch_size=4),stats,'cuda')
    pose_loss=CartesianPoseLoss(fk_spec_from_scene(scene),stats).cuda()
    before=evaluate_response(policy,stats,manifest,checkpoint_sha256=plan['initializer_sha256'])
    (output/'response_before.json').write_text(json.dumps(before,indent=2)+'\n')
    print(json.dumps(dict(stage='response_before',passed=before['passed'],
        direction=before['direction_pass_fraction'],error_m=before['maximum_prefix_position_error_m'])),flush=True)
    frozen={n:p.detach().cpu().clone() for n,p in policy.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    rates=[g['lr'] for g in groups];started=time.monotonic();artifacts=[]
    loader=DataLoader(dataset,batch_sampler=batches.tolist(),num_workers=2,multiprocessing_context='spawn',pin_memory=True)
    for step,(image,current,target,padding,_) in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        prediction=policy(current.cuda(),image.cuda())
        task,parts=executed_objective(prediction,target.cuda(),padding.cuda(),stats,pose_loss)
        preservation=keeper.loss(policy,1000+step-1)
        weight,norms=preservation_scale(task,preservation,policy.base.model.input_proj_robot_state.parameters())
        loss=task+weight*preservation
        if not torch.isfinite(loss):raise ValueError('Nonfinite continuation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        for group,rate in zip(optimizer.param_groups,rates):group['lr']=rate*(1-.9*(step-1)/2999)
        optimizer.step()
        if step==1 or step%100==0:
            record=dict(phase='training',updates=step,limit=3000,cumulative_new_optimizer_updates=3000+step,
                task=float(task.detach()),preservation=float(preservation.detach()),preservation_weight=weight,
                weighted_preservation_gradient_fraction=weight*norms['preservation_gradient_norm']/max(norms['task_gradient_norm'],1e-12),
                terms={k:float(v.detach()) for k,v in parts.items()},elapsed_s=time.monotonic()-started)
            temporary=output/'status.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n')
            temporary.replace(output/'status.json');print(json.dumps(record),flush=True)
        if step%1000==0:
            if any(not torch.equal(p.detach().cpu(),frozen[n]) for n,p in policy.named_parameters() if n in frozen):
                raise ValueError('Frozen visual/CVAE parameters changed')
            require_unchanged(hashes)
            name='policy_last.ckpt' if step==3000 else f'policy_step_{step}.ckpt'
            checkpoint=output/name
            torch.save(dict(profile=PROFILE,model=policy.state_dict(),stats=stats,config=policy.config,
                actual_updates=step,cumulative_new_optimizer_updates=3000+step,
                optimizer=optimizer.state_dict(),optimizer_initialization='new_adamw_from_latest_weights',
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                hardware_deployment_allowed=False),checkpoint)
            gate=evaluate_response(policy,stats,manifest,checkpoint_sha256=snapshot([checkpoint])[str(checkpoint.resolve())])
            gate_path=output/('response_gate.json' if step==3000 else f'response_step_{step}.json')
            gate_path.write_text(json.dumps(gate,indent=2)+'\n');artifacts += [checkpoint,gate_path]
            print(json.dumps(dict(stage='response',updates=step,passed=gate['passed'],
                direction=gate['direction_pass_fraction'],error_m=gate['maximum_prefix_position_error_m'])),flush=True)
    if step!=3000:raise ValueError('Exactly3000 continuation updates required')
    require_unchanged(hashes)
    proof=dict(complete=True,actual_updates=3000,formal_epochs=0,initializer_new_updates=3000,
        cumulative_new_optimizer_updates=6000,source_sha256=hashes,checkpoint_sha256=snapshot(artifacts),
        frozen_parameters_unchanged=True,optimizer_initialization='new_adamw_from_latest_weights',
        position_response_gate_passed=gate['passed'],full_task_passed=False,
        hardware_commands_sent=False,hardware_deployment_allowed=False)
    (output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    record.update(phase='complete',position_response_gate_passed=gate['passed'])
    (output/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(stage='continuation_complete',updates=3000,response_passed=gate['passed'])),flush=True)


if __name__=='__main__':main()
