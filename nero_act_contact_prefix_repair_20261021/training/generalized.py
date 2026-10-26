"""Explicit48-episode, fresh-statistics ACT training; never imports ROS or SDK."""
import argparse,hashlib,json,math,pickle,sys
from functools import partial
from pathlib import Path
import h5py
import numpy as np
import torch
from training.train import FullTaskPolicy,complete_windows,validation_windows,save_checkpoint,training_reach_deadline,trainer
from training.command_context import ReferenceDataset
from training.rebase_initializer import paired_config,rebase_initializer
from training.integrity import snapshot,require_unchanged
from src.act.nero_dataset import compute_norm_stats

ROOT=Path(__file__).resolve().parents[1]


def validate_manifest(manifest,split,*,check_files=True):
    rows=manifest['episodes'];train,val=split['train'],split['val']
    ids=[r.get('id') for r in rows];groups=[r.get('source_group') for r in rows]
    resolved=[str(Path(r['path']).resolve()) for r in rows]
    if (len(rows)!=48 or len(train)!=36 or len(val)!=12 or len(set(ids))!=48
            or set(train)&set(val) or set(ids)!=set(train+val) or len(set(groups))!=48
            or any(g is None for g in groups) or len(set(resolved))!=48):
        raise ValueError('48 independent verified scenes with disjoint36/12 splits required')
    paths=[]
    for row in rows:
        expected='train' if row['id'] in train else 'val'
        if (row.get('status')!='valid' or row.get('full_task_verified') is not True
                or row.get('teacher_success') is not True or row.get('split')!=expected
                or row.get('replay',{}).get('passed') is not True
                or row.get('prefix_contract_audit',{}).get('passed') is not True):
            raise ValueError('Incomplete, rejected or unaudited demonstration cannot enter training')
        path=Path(row['path'])
        if path.name!=f"episode_{row['id']}.hdf5":
            raise ValueError('Filename-derived episode ID must equal manifest and recorded scene ID')
        if check_files:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row.get('sha256'):
                raise ValueError('Physical dataset file changed: '+str(path))
            with h5py.File(path,'r') as f:
                if (f.attrs.get('task')!='complete_pick_carry_place_retreat' or not f.attrs.get('full_task_verified')
                        or not f.attrs.get('teacher_success') or not f.attrs.get('sim')):
                    raise ValueError('Physical demonstration roles disagree with manifest')
                recorded=json.loads(f.attrs['scene_config'])['generalization']
                if recorded['index']!=row['id'] or recorded['source_group']!=row['source_group'] or recorded['split']!=expected:
                    raise ValueError('Recorded independent scene and split disagree with manifest')
        paths.append(path)
    return paths


def combined_inputs(manifest,split,corrections=None,*,check_files=True):
    paths=validate_manifest(manifest,split,check_files=check_files)
    result=dict(manifest,episodes=list(manifest['episodes']))
    partition={key:list(value) for key,value in split.items()}
    if corrections is not None:
        from training.recoveries import validate_corrections
        batches=corrections if isinstance(corrections,list) else [corrections]
        if not batches:raise ValueError('Nonempty correction batches required')
        for batch in batches:
            paths+=validate_corrections(batch,result['episodes'],check_files=check_files)
            result['episodes']+=batch['episodes']
            for row in batch['episodes']:partition[row['split']].append(row['id'])
        result['correction_source_checkpoint_sha256']=batches[0]['source_checkpoint_sha256']
        result['old_evaluation_is_development_only']=True
    return result,partition,paths


def bind_independent_cases(path):
    from simulation.evaluate_generalized import load_predeclared_cases
    config=dict(independent_cases_file=str(path.resolve()),
        independent_cases_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    load_predeclared_cases(config)
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--initializer',type=Path,default=ROOT/'initializers/run13/policy.ckpt')
    parser.add_argument('--corrections',type=Path)
    parser.add_argument('--additional-corrections',type=Path,action='append',default=[])
    parser.add_argument('--independent-cases',type=Path)
    parser.add_argument('--train-visual-backbone',action='store_true')
    parser.add_argument('--dense-contact-inputs',action='store_true')
    parser.add_argument('--pose-objective',action='store_true')
    parser.add_argument('--cvae-kl-weight',type=float)
    parser.add_argument('--prior-gripper-weight',type=float)
    parser.add_argument('--train-gripper-head-only',action='store_true')
    parser.add_argument('--train-nonlinear-gripper-only',action='store_true')
    parser.add_argument('--gripper-adapter-lr',type=float,default=.001)
    parser.add_argument('--balance-gripper-transitions',action='store_true')
    parser.add_argument('--focus-grasp-recoveries',action='store_true')
    parser.add_argument('--run',required=True);parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args()
    if Path(args.run).name!=args.run or not 1<=args.epochs<=20:parser.error('New run name and1..20 declared epochs required')
    if args.cvae_kl_weight is not None and (not math.isfinite(args.cvae_kl_weight) or args.cvae_kl_weight<=0):
        parser.error('CVAE training requires finite positive KL regularization')
    if args.prior_gripper_weight is not None and (not math.isfinite(args.prior_gripper_weight) or args.prior_gripper_weight<0):
        parser.error('Prior gripper supervision requires a finite nonnegative weight')
    if args.train_gripper_head_only and args.train_visual_backbone:
        parser.error('Gripper-only adaptation cannot train the visual backbone')
    if args.train_nonlinear_gripper_only and (args.train_visual_backbone or args.train_gripper_head_only
            or not math.isfinite(args.gripper_adapter_lr) or args.gripper_adapter_lr<=0
            or args.prior_gripper_weight is not None and args.prior_gripper_weight<=0):
        parser.error('Nonlinear adapter-only training requires a positive rate and prior loss, with other training modes disabled')
    if args.balance_gripper_transitions and (not args.train_nonlinear_gripper_only or not args.dense_contact_inputs):
        parser.error('Balanced gripper sampling requires nonlinear adapter-only and dense contact inputs')
    if args.focus_grasp_recoveries and not args.balance_gripper_transitions:
        parser.error('Focused direct grasp recovery sampling requires balanced adapter mode')
    data=args.data_dir.resolve();manifest=json.loads((data/'manifest.json').read_text());split=json.loads((data/'split.json').read_text())
    correction_manifest=json.loads(args.corrections.read_text()) if args.corrections else None
    if args.additional_corrections and correction_manifest is None:parser.error('Additional corrections require a primary batch')
    correction_batches=([correction_manifest]+[json.loads(p.read_text()) for p in args.additional_corrections]
        if correction_manifest else None)
    if correction_manifest and args.independent_cases is None:parser.error('Corrections require a predeclared independent case file')
    independent=bind_independent_cases(args.independent_cases) if args.independent_cases else {}
    manifest,split,paths=combined_inputs(manifest,split,correction_batches)
    observation_sources=[]
    if args.dense_contact_inputs:
        from simulation.refresh_observations import require_fresh_inputs
        observation_sources=require_fresh_inputs(paths)
    if correction_manifest and hashlib.sha256(args.initializer.read_bytes()).hexdigest()!=correction_manifest['source_checkpoint_sha256']:
        raise ValueError('This bounded correction run must initialize from the diagnosed frozen ACT')
    train_paths=[p for p,row in zip(paths,manifest['episodes']) if row['id'] in split['train']]
    stats=compute_norm_stats(train_paths)
    source=torch.load(args.initializer,map_location='cpu',weights_only=False)
    old={k:np.array(v,np.float32) for k,v in source['config']['constraint_loss_stats'].items()}
    config=paired_config(source['config'],stats)
    config.update(independent)
    config['freeze_backbone']=not args.train_visual_backbone
    config['train_gripper_head_only']=args.train_gripper_head_only
    config['train_nonlinear_gripper_only']=args.train_nonlinear_gripper_only
    config['balance_gripper_transitions']=args.balance_gripper_transitions
    config['focus_grasp_recoveries']=args.focus_grasp_recoveries
    config['trainable_parameter_scope']=('full_policy' if args.train_visual_backbone
                                         else 'policy_with_frozen_visual_backbone')
    output=ROOT/'training'/args.run;output.mkdir(exist_ok=False)
    (output/'artifacts').mkdir()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'split.json').write_text(json.dumps(split,indent=2))
    stats_path=output/'training_only_stats.pkl'
    with stats_path.open('wb') as stream:pickle.dump(stats,stream)
    config.update(dataset_dir=str(data),manifest=str(output/'manifest.json'),split_file=str(output/'split.json'),
        checkpoint_dir=str(output/'checkpoints'),artifact_dir=str(output/'artifacts'),training_log=str(output/'training.csv'),
        normalization_stats=str(stats_path),num_epochs=args.epochs,device='cuda',cpu_threads=2,
        reset_action_head=False,normalization_source='only_new_training_scene_groups',
        training_integrity_required=True,
        scene_config=json.loads((data/'scene_config.json').read_text()),
        reach_deadline_steps=training_reach_deadline(paths,split['train']),
        training_revision='position_generalization_causal_context_v1',hardware_deployment_allowed=False)
    if correction_manifest:
        config.update(training_revision='position_generalization_first_lift_recovery_v1',
            correction_manifest=str(args.corrections.resolve()),
            additional_correction_manifests=[str(p.resolve()) for p in args.additional_corrections],
            correction_role='verified_partial_recoveries_not_full_demonstrations',
            prior_evaluation_scope='development_regression_after_failed_state_corrections')
    if args.train_visual_backbone:
        config.update(training_revision='position_generalization_visual_learning_v1',
            visual_learning_control='same54data_run14_initializer_seed42_20epochs_only_visual_training_changed')
    if args.dense_contact_inputs:
        config.update(training_revision='position_generalization_dense_contact_inputs_v1',
            critical_input_sampling=dict(stages=['close','lift','release'],first_points=20,
                lift_first5_repeats=6,lift_later_repeats=2,other_added_repeats=1,original_wait_filters_preserved=True,
                descend_recovery_all_inputs_20hz=True),
            visual_learning_control='visual_training_and_dense_critical_inputs_with_verified_alignment_recoveries',
            critical_rgb_source='fresh_preaction_full_physics_renders; identical_actions_states_scene')
    if args.pose_objective:
        from training.pose_objective import fk_spec_from_scene
        config.update(pose_loss_weight=.05,pose_kinematics_spec=fk_spec_from_scene(config['scene_config']),
            pose_loss_scales=dict(position_m=.01,rotation_chord_rad=float(np.deg2rad(1))),
            checkpoint_selection='minimum_validation_total_joint_and_cartesian_pose_loss',
            training_revision='position_generalization_cartesian_pose_objective_v1')
    if args.cvae_kl_weight is not None:
        config.update(cvae_training=True,kl_weight=args.cvae_kl_weight,
            cvae_validation='zero_latent_no_future_actions',
            checkpoint_selection='minimum_zero_latent_validation_total_loss',
            training_revision='position_generalization_cvae_posterior_training_v1')
    if args.prior_gripper_weight is not None:
        config.update(prior_gripper_loss_weight=args.prior_gripper_weight,
            checkpoint_selection='minimum_zero_latent_validation_total_with_prior_gripper_loss',
            training_revision='position_generalization_prior_gripper_gradient_v1')
    if args.train_gripper_head_only:
        config.update(train_gripper_head_only=True,weight_decay=0.,freeze_backbone=True,
            trainable_parameter_scope='only_gripper_row_of_action_head_weight_and_bias',
            visual_learning_control='frozen learned visual and arm network; only gripper output row adapts',
            training_revision='position_generalization_gripper_head_only_v1')
    if args.train_nonlinear_gripper_only:
        config.update(nonlinear_gripper_head=True,train_nonlinear_gripper_only=True,
            gripper_adapter_lr=args.gripper_adapter_lr,freeze_backbone=True,weight_decay=0.,
            prior_gripper_loss_weight=(args.prior_gripper_weight
                if args.prior_gripper_weight is not None else .05),
            trainable_parameter_scope='only_nonlinear_gripper_adapter',
            cvae_adaptation_scope='pretrained_posterior_frozen; adapter_uses_zero_latent_in_training_and_inference',
            checkpoint_selection='minimum_zero_latent_validation_total_with_prior_gripper_loss',
            training_revision='position_generalization_nonlinear_gripper_adapter_v1')
    if args.balance_gripper_transitions:
        config.update(gripper_sampling='equal_mass_first20_close_first20_release_background_seed42',
            checkpoint_selection='minimum_zero_latent_balanced_validation_total',
            training_revision='position_generalization_balanced_gripper_adapter_v1')
        config['critical_input_sampling']['repeat_metadata_scope']='candidate_generator_only; repeats_discarded_before_equal_role_sampling'
        if args.focus_grasp_recoveries:
            config.update(gripper_sampling='equal_four_roles_direct_grasp_close_other_close_release_background_seed42',
                training_revision='position_generalization_focused_gripper_adapter_v1')
    init=output/'artifacts/rebased_initializer.ckpt';config['init_checkpoint']=str(init)
    torch.save(dict(model=rebase_initializer(source['model'],old,stats,config),config=config,
        initialization_only=True,hardware_deployment_allowed=False,
        source_checkpoint_sha256=hashlib.sha256(args.initializer.read_bytes()).hexdigest()),init)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'artifacts/data_sha256.json').write_text(json.dumps(snapshot(paths),indent=2))
    sources=[p for folder in ['training','executor','simulation','policy_source'] for p in (ROOT/folder).rglob('*.py')]
    inputs=paths+sources+[data/'manifest.json',data/'split.json',data/'scene_config.json',
        args.initializer,init,stats_path,output/'config.json',output/'manifest.json',output/'split.json',ROOT/'scene/center.json']
    if args.corrections:
        for file,batch in zip([args.corrections,*args.additional_corrections],correction_batches):
            origin=Path(batch['source_evaluation'])
            inputs+=[file,origin/'protocol.json']
            for row in batch['episodes']:
                inputs+=[Path(row['origin'])/name for name in ('physical_snapshots.json','scene_configuration.json')]
    if args.independent_cases:inputs.append(args.independent_cases)
    inputs+=observation_sources
    hashes=snapshot(inputs)
    (output/'artifacts/training_inputs_before.json').write_text(json.dumps(hashes,indent=2))
    trainer.NeroActPolicy=FullTaskPolicy;trainer.NeroEpisodicDataset=ReferenceDataset
    trainer.fixed_window_samples=partial(complete_windows,command_context=True)
    trainer.fixed_validation_samples=partial(validation_windows,command_context=True)
    if args.dense_contact_inputs:
        from training.transition_sampling import critical_windows,critical_validation_windows
        trainer.fixed_window_samples=partial(critical_windows,command_context=True)
        trainer.fixed_validation_samples=partial(critical_validation_windows,command_context=True)
    if args.balance_gripper_transitions:
        from training.transition_sampling import balanced_gripper_windows,balanced_gripper_validation_windows
        trainer.fixed_window_samples=partial(balanced_gripper_windows,command_context=True,
            focus_grasp_recovery=args.focus_grasp_recoveries)
        trainer.fixed_validation_samples=partial(balanced_gripper_validation_windows,command_context=True,
            focus_grasp_recovery=args.focus_grasp_recoveries)
    trainer.save_checkpoint=save_checkpoint
    torch.set_num_threads(2);sys.argv=['retained_act_trainer','--config',str(output/'config.json')]
    result=trainer.main()
    require_unchanged(hashes)
    summary=json.loads((output/'artifacts/training_summary.json').read_text())
    if summary['last_metrics']['epoch']!=args.epochs:raise RuntimeError('Declared optimization epochs did not complete')
    proof=dict(finalized=True,declared_epochs=args.epochs,actual_epochs=summary['last_metrics']['epoch'],
        protected_sha256=hashes,output_sha256=snapshot(list((output/'checkpoints').glob('policy*.ckpt'))+
            [output/'checkpoints/dataset_stats.pkl',output/'artifacts/training_summary.json']))
    (output/'artifacts/training_integrity.json').write_text(json.dumps(proof,indent=2))
    return result


if __name__=='__main__':raise SystemExit(main())
