"""Adapt only the existing gripper head of a finalized visual15 ACT checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from executor.contract import VISUAL_PROFILE
from executor.model import verify_paired_readout_statistics
from training.contact_sampling import PROFILE,SELECTION_PROFILE
from training.integrity import require_finalized,snapshot
from training.train import FullTaskPolicy,save_checkpoint
from training.visual import visual_windows,run_epoch,complete_training
from training.visual_dataset import VisualReferenceDataset,validate_visual_groups

ROOT=Path(__file__).resolve().parents[1]


def build_contact_adapter(config):
    if config.get('visual_observation_profile')!=VISUAL_PROFILE or config.get('policy_input_dim')!=15:
        raise ValueError('An already trained visual15 policy is required')
    c=dict(config,nonlinear_gripper_head=True,freeze_backbone=True,
        train_nonlinear_gripper_only=True,train_arm_output_and_gripper_only=False,
        train_gripper_head_only=False,reset_action_head=False,gripper_adapter_lr=.0003)
    policy=FullTaskPolicy(c);policy.visual_gripper_repair=True
    if any(p.requires_grad != n.startswith('gripper_adapter.') for n,p in policy.named_parameters()):
        raise RuntimeError('Only the existing gripper adapter may receive optimization')
    return policy


def train(data_dir,initializer,output):
    data_dir,initializer,output=map(lambda p:Path(p).resolve(),[data_dir,initializer,output])
    if output.exists():raise FileExistsError(output)
    source=torch.load(initializer,map_location='cpu',weights_only=False)
    finalized=require_finalized(initializer,source['config'])
    split=json.loads((data_dir/'split.json').read_text());manifest=json.loads((data_dir/'manifest.json').read_text())
    if split!=source['split'] or manifest.get('visual_sampling_profile')!=PROFILE or not manifest.get('complete'):
        raise ValueError('Same48 source groups and completed repaired image pool required')
    paths={role:[Path(r['path']) for r in manifest['episodes'] if r['id'] in split[role]] for role in ['train','val']}
    for row in manifest['episodes']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Derived physical labels/images changed')
    groups=validate_visual_groups(paths['train'],paths['val'])
    stats_path=initializer.parent/'dataset_stats.pkl'
    with stats_path.open('rb') as f:stats=pickle.load(f)
    config=dict(source['config'],device='cuda',num_epochs=20,batch_size=32,cpu_threads=2,data_workers=2,
        pretrained_backbone=False,freeze_backbone=True,train_nonlinear_gripper_only=True,
        train_arm_output_and_gripper_only=False,train_gripper_head_only=False,reset_action_head=False,
        gripper_adapter_lr=.0003,visual_sampling_profile=PROFILE,visual_selection_profile=SELECTION_PROFILE,
        training_revision='visual15_existing_gripper_only_force_building_repair_v1',
        checkpoint_dir=str(output/'checkpoints'),artifact_dir=str(output/'artifacts'),
        training_log=str(output/'training.csv'),dataset_dir=str(data_dir),
        statistics_source='unchanged_paired_statistics_of_finalized_same36_training_groups',
        frozen_trained_CVAE_preserved=True,hardware_deployment_allowed=False)
    verify_paired_readout_statistics(config,stats)
    if config['calibration_sha256']!=manifest['calibration_sha256']:raise ValueError('Calibration changed')
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output.mkdir(parents=True);(output/'checkpoints').mkdir();(output/'artifacts').mkdir()
    (output/'config.json').write_text(json.dumps(config,indent=2));(output/'split.json').write_text(json.dumps(split,indent=2))
    with (output/'checkpoints/dataset_stats.pkl').open('wb') as f:pickle.dump(stats,f)
    policy=build_contact_adapter(config).cuda();policy.load_state_dict(source['model']);policy.declared_training_epochs=20
    frozen={n:p.detach().cpu().clone() for n,p in policy.state_dict().items() if not n.startswith('gripper_adapter.')}
    files=[p for name in ['training','executor','calibration','vision','simulation','policy_source'] for p in (ROOT/name).rglob('*.py')]
    protected=snapshot(files+[p for role in paths.values() for p in role]+[
        initializer,stats_path,finalized,data_dir/'manifest.json',data_dir/'split.json',
        data_dir/'calibration_profile.json',data_dir/'scene_config.json',ROOT/'evaluation/protocol.json',
        ROOT/'evidence/baseline_development_protocol.json',output/'config.json',output/'checkpoints/dataset_stats.pkl'])
    (output/'artifacts/training_inputs_before.json').write_text(json.dumps(protected,indent=2))
    loaders={}
    for role in paths:
        samples=visual_windows(paths[role])
        if role=='val':samples=list(dict.fromkeys(samples))
        ds=VisualReferenceDataset(paths[role],config['camera_names'],stats,policy_config=config,
            image_size=config['image_size'],sample_starts=samples)
        loaders[role]=DataLoader(ds,batch_size=32,shuffle=role=='train',num_workers=2,pin_memory=True)
    (output/'artifacts/data_groups.json').write_text(json.dumps(dict(groups=groups,
        train_windows=len(loaders['train'].dataset),val_windows=len(loaders['val'].dataset)),indent=2))
    history=[];best=float('inf');started=time.monotonic();writer=None
    with (output/'training.csv').open('w') as stream:
        for epoch in range(1,21):
            metrics=run_epoch(policy,loaders['train'],torch.device('cuda'),training=True,output=output,epoch=epoch)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(42);val=run_epoch(policy,loaders['val'],torch.device('cuda'),training=False)
            for n,p in frozen.items():
                if not torch.equal(policy.state_dict()[n].detach().cpu(),p):raise RuntimeError('Frozen trained arm/encoder/CVAE changed: '+n)
            row=dict(epoch=epoch,**{'train_'+k:v for k,v in metrics.items()},**{'val_'+k:v for k,v in val.items()},elapsed_s=time.monotonic()-started)
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);stream.flush();history.append(row)
            save_checkpoint(output/'checkpoints/policy_last.ckpt',policy,config,split,epoch,row)
            if val['loss']<best:best=val['loss'];save_checkpoint(output/'checkpoints/policy_best_val.ckpt',policy,config,split,epoch,row)
            print(json.dumps(row),flush=True)
    (output/'artifacts/frozen_parameters_verified.json').write_text(json.dumps(dict(
        unchanged_non_gripper_tensors=len(frozen),verified_every_epoch=True,paired_statistics_unchanged=True),indent=2))
    summary=dict(complete=True,actual_epochs=20,best_zero_latent_val_loss=best,elapsed_s=time.monotonic()-started,
        frozen_trained_base_unchanged=True,hardware_deployment_allowed=False)
    complete_training(output,config,history,protected,summary)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--initializer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    train(a.data_dir,a.initializer,a.output)
