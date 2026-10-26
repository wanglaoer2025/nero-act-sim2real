"""Complete-task training adapter around the retained ACT model and trainer."""
import argparse
import hashlib
import json
import math
import pickle
from functools import partial
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'policy_source'))
from src.act import train as trainer
from src.act.nero_dataset import compute_norm_stats
from executor.model import BoundedActPolicy
from executor.contract import GRIPPER_CONTEXT,GRIPPER_CONTEXT_SCALE_M


def weighted_l1(prediction, target, padding):
    valid = (~padding[:, :20]).to(prediction.dtype)
    weights = valid.clone()
    weights[:, :10] *= 5
    return ((prediction[:, :20]-target[:, :20]).abs()*weights[:, :, None]).sum() / (
        weights.sum()*8).clamp_min(1)


def prefix_margin_penalty(prediction, qpos, padding, stats):
    values = {key: torch.as_tensor(value, device=prediction.device, dtype=prediction.dtype)
              for key, value in stats.items()}
    action = prediction[:, :10]*values['action_std']+values['action_mean']
    actual = qpos[:,:8]*values['qpos_std']+values['qpos_mean']
    margin = prediction.new_tensor([.025]*7+[.008])
    excess = torch.relu((action-actual[:, None]).abs()-margin)/values['action_std']
    valid = (~padding[:, :10]).to(prediction.dtype)
    return (excess*valid[:, :, None]).sum()/(valid.sum()*8).clamp_min(1)


def training_statistics(paths, train_indices):
    if not train_indices or len(set(train_indices)) != len(train_indices):
        raise ValueError('Unique nonempty training split required')
    return compute_norm_stats([paths[index] for index in train_indices])


def reuse_training_statistics(checkpoint, train_episode_ids):
    bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not set(bundle['split']['train']).issubset(set(train_episode_ids)):
        raise ValueError('Frozen statistics were not computed solely from the new training split')
    with (Path(checkpoint).parent/'dataset_stats.pkl').open('rb') as f:
        return pickle.load(f)


def complete_windows(paths, chunk_size, *, stride=10, command_context=False, **kwargs):
    if chunk_size != 20 or stride != 10:
        raise ValueError('Complete-task contract requires20-step chunks and10-step stride')
    samples = []
    for index, path in enumerate(paths):
        with h5py.File(path, 'r') as f:
            stages = f['simulation/stage'].asstr()[:]
            correction = str(f.attrs.get('task', '')).endswith('_correction')
            actions = f['action'][:]
            widths = f['observations/gripper_state'][:, 0]
        for start in range(0, len(stages), stride):
            end = min(start+10, len(stages))
            stationary = np.max(abs(actions[start:end, :7]-actions[start, :7])) <= 1e-6
            contact_wait = stages[start] == 'close' and widths[start]-actions[start, 7] >= .003
            open_wait = stages[start] == 'release' and widths[start] > .09 and abs(widths[start]-actions[start, 7]) <= .001
            release_deadband = stages[start] == 'release' and actions[start, 7] < widths[start]-.0005
            # A feedback-only policy cannot learn elapsed-time waits at an unchanged state.
            if stationary and (contact_wait or open_wait or (release_deadband and not command_context)):
                continue
            repeat = 3 if correction or stages[start] in ('close', 'release', 'settle') else 1
            samples.extend([(index, start)]*repeat)
    return samples


def validation_windows(paths, samples_per_episode, chunk_size, *args, **kwargs):
    samples = list(dict.fromkeys(complete_windows(paths, chunk_size,command_context=kwargs.get('command_context',False))))
    records = [dict(episode_id=int(paths[i].stem.removeprefix('episode_')), start_ts=t)
               for i, t in samples]
    return samples, records


class FullTaskPolicy(BoundedActPolicy):
    def __init__(self, config):
        super().__init__(config)
        if config.get('freeze_backbone',True):
            for backbone in self.model.backbones:
                for parameter in backbone.parameters():
                    parameter.requires_grad_(False)
        self.reset_head = bool(config.get('reset_action_head', False))
        self.noise = torch.tensor(config.get('training_observation_noise_normalized', [0]*8))
        self.margin_weight = float(config.get('prefix_margin_loss_weight', 0.))
        self.loss_stats = config.get('constraint_loss_stats')

    def load_state_dict(self, state_dict, strict=True):
        result = super().load_state_dict(state_dict, strict=strict)
        if self.reset_head:
            with torch.no_grad():
                self.model.action_head.weight.zero_()
                self.model.action_head.bias.zero_()
            self.reset_head = False
        return result

    def forward(self, qpos, image, actions=None, is_pad=None):
        if actions is not None and self.training:
            noise=(torch.rand_like(qpos[:,:8])*2-1)*self.noise.to(qpos.device)
            if self.gripper_context:
                qpos=torch.cat([qpos[:,:8]+noise,qpos[:,8:9]-noise[:,7:8]*self.feedback_qpos_std[7]/GRIPPER_CONTEXT_SCALE_M],dim=1)
            else:qpos=qpos+noise
        prediction = super().forward(qpos, image)
        if actions is None:
            return prediction
        l1 = weighted_l1(prediction, actions, is_pad)
        penalty = (prefix_margin_penalty(prediction, qpos, is_pad, self.loss_stats)
                   if self.margin_weight else prediction.sum()*0)
        loss = l1+self.margin_weight*penalty
        zero = prediction.sum()*0
        return dict(loss=loss, l1=l1, kl=zero, gripper_bce=zero, prefix_margin=penalty, a_hat=prediction)


def save_checkpoint(path, policy, config, split, epoch, metrics):
    torch.save(dict(model=policy.state_dict(), config=config, split=split, epoch=epoch,
                    metrics=metrics, optimizer_omitted_for_inference=True,
                    hardware_deployment_allowed=False,
                    handoff_scope='complete-task simulation training; hardware unverified'), path)


def dataset_scene(data_dir):
    data_dir = Path(data_dir).resolve()
    scene_file = (ROOT/'scene/config.json' if data_dir == ROOT/'training_data'
                  else data_dir/'scene_config.json')
    return json.loads(scene_file.read_text())


def check_initializer_dataset(checkpoint, data_dir, scene):
    if Path(data_dir).resolve() == ROOT/'training_data':
        return
    config = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
    if (Path(config.get('dataset_dir', '')).resolve() != Path(data_dir).resolve()
            or config.get('scene_config') != scene):
        raise ValueError('Changed dataset/scene requires fresh statistics and reset action head')


def training_reach_deadline(paths, train_ids):
    lengths = []
    for path in paths:
        if int(Path(path).stem.removeprefix('episode_')) in train_ids:
            with h5py.File(path, 'r') as f:
                if str(f.attrs.get('task', '')).endswith('_correction'):
                    continue
                lengths.append(len(f['action']))
    if not lengths:
        raise ValueError('Training-only task duration evidence required')
    deadline = max(3200, math.ceil(max(lengths)*1.25/100)*100)
    if deadline > 6000:
        raise ValueError('New task duration exceeds this experimental profile')
    return deadline


def supplemental_full_inputs(manifest_path, used_ids):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['rejected']:
        raise ValueError('Rejected full episodes cannot enter training')
    used_ids = set(used_ids)
    for item in manifest['episodes']:
        path = Path(item['path'])
        episode_id = int(path.stem.removeprefix('episode_'))
        if (item.get('status') != 'valid' or item.get('full_task_verified') is not True
                or item.get('split') not in ('train', 'val') or episode_id < 12 or episode_id in used_ids):
            raise ValueError('Separate verified full-task IDs and explicit splits required')
        with h5py.File(path, 'r') as f:
            if (not f.attrs.get('sim') or not f.attrs.get('teacher_success')
                    or not f.attrs.get('full_task_verified')
                    or f.attrs.get('task') != 'complete_pick_carry_place_retreat'):
                raise ValueError('Supplemental full-task data lacks complete physical acceptance')
        used_ids.add(episode_id)
    return manifest['episodes']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--run', default='run0')
    parser.add_argument('--corrections', type=Path)
    parser.add_argument('--init-checkpoint', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'training_data')
    parser.add_argument('--supplemental-full', type=Path)
    parser.add_argument('--prefix-margin-loss', action='store_true')
    parser.add_argument('--bounded-feedback-head', action='store_true')
    parser.add_argument('--previous-gripper-context',action='store_true')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 200 or Path(args.run).name != args.run:
        parser.error('Invalid epochs or run name')
    if args.previous_gripper_context and not args.bounded_feedback_head:
        parser.error('Previous-gripper context requires the trained bounded feedback head')
    args.data_dir = args.data_dir.resolve()
    scene = dataset_scene(args.data_dir)
    if args.init_checkpoint is not None:
        check_initializer_dataset(args.init_checkpoint, args.data_dir, scene)
    output = ROOT/'training'/args.run
    output.mkdir(exist_ok=False)
    manifest = json.loads((args.data_dir/'manifest.json').read_text())
    if manifest['rejected'] or len(manifest['episodes']) != 12:
        raise ValueError('Twelve accepted complete physical episodes required')
    paths = [args.data_dir/f'episode_{i}.hdf5' for i in range(12)]
    split = dict(train=list(range(8)), val=[8, 9])
    extra = []
    if args.corrections is not None:
        corrections = json.loads(args.corrections.read_text())
        if corrections['rejected']:
            raise ValueError('Rejected correction episodes cannot enter training')
        for item in corrections['episodes']:
            from simulation.teacher import CONTINUATION_ROLES
            if (item['status'] != 'valid' or item['task_role'] not in CONTINUATION_ROLES.values()
                    or item['full_task_verified'] is not False or item['correction_verified'] is not True
                    or item['split'] not in ('train', 'val')):
                raise ValueError('Explicit verified partial-correction role/split required')
            path = Path(item['path'])
            with h5py.File(path, 'r') as f:
                if f.attrs.get('task') != item['task_role'] or f.attrs.get('full_task_verified'):
                    raise ValueError('Correction HDF role/full-task label mismatch')
            episode_id = int(path.stem.removeprefix('episode_'))
            if episode_id < 12 or episode_id in split['train']+split['val']:
                raise ValueError('Correction episode IDs must be unique and separate from base data')
            split[item['split']].append(episode_id)
            paths.append(path)
            extra.append(item)
    if args.supplemental_full is not None:
        for item in supplemental_full_inputs(args.supplemental_full, list(range(12))+split['train']+split['val']):
            path = Path(item['path'])
            split[item['split']].append(int(path.stem.removeprefix('episode_')))
            paths.append(path)
            extra.append(item)
    from simulation.collect import verify_action_contract
    for path in paths:
        verify_action_contract(path)
    stats = (reuse_training_statistics(args.init_checkpoint, split['train'])
             if args.init_checkpoint is not None else
             training_statistics(paths, [i for i, p in enumerate(paths)
                                         if int(p.stem.removeprefix('episode_')) in split['train']]))
    stats_path = output/'training_only_stats.pkl'
    with stats_path.open('wb') as f:
        pickle.dump(stats, f)
    train_manifest = dict(episodes=manifest['episodes'][:10]+extra)
    (output/'manifest.json').write_text(json.dumps(train_manifest, indent=2))
    (output/'split.json').write_text(json.dumps(split, indent=2))
    config = json.loads((ROOT.parent/'nero_cup_retrained_v2_20261016/model/training_config.json').read_text())
    config.update(manifest=str(output/'manifest.json'), dataset_dir=str(args.data_dir),
                  split_file=str(output/'split.json'), checkpoint_dir=str(output/'checkpoints'),
                  artifact_dir=str(output/'artifacts'), training_log=str(output/'training.csv'),
                  normalization_stats=str(stats_path), device='cuda', cpu_threads=2,
                  num_epochs=args.epochs, reset_action_head=args.init_checkpoint is None, tcp_position_loss_weight=0,
                  init_checkpoint=str(args.init_checkpoint.resolve() if args.init_checkpoint is not None else
                                      ROOT.parent/'nero_cup_retrained_v2_20261016/model/policy.ckpt'),
                  task_scope='complete_pick_carry_place_retreat',
                  bounded_gripper_output=True,
                  gripper_action_mean=float(stats['action_mean'][7]),
                  gripper_action_std=float(stats['action_std'][7]),
                  training_revision='full_task_no_stationary_contact_prefix',
                  stationary_contact_prefixes_excluded=True,
                  release_deadband_prefixes_excluded=True,
                  stage_sampling=dict(close_repeats=3, release_repeats=3, settle_repeats=3, corrective_repeats=3),
                  normalization_source=('frozen_training_statistics_from_initializer' if args.init_checkpoint is not None else 'all_training_episodes'),
                  joint_anchor_scale=(stats['qpos_std'][:7]/stats['action_std'][:7]).tolist(),
                  joint_anchor_offset=((stats['qpos_mean'][:7]-stats['action_mean'][:7])/stats['action_std'][:7]).tolist(),
                  training_observation_noise_normalized=(np.r_[np.full(7, .0002), .0001]/stats['qpos_std']).tolist())
    config['scene_config'] = scene
    config.update(state_dim=8,physical_state_dim=8,policy_input_dim=9 if args.previous_gripper_context else 8,
        gripper_command_context=GRIPPER_CONTEXT if args.previous_gripper_context else None,
        coordinate_conditioning=False,previous_gripper_conditioning=False,task_displacement_features=False)
    config['release_deadband_prefixes_excluded']=not args.previous_gripper_context
    config['reach_deadline_steps'] = training_reach_deadline(paths, split['train'])
    config['task_budget_rule'] = 'max3200_or125percent_longest_training_episode_rounded100; final100hold_separate'
    config['prefix_margin_loss_weight'] = 1. if args.prefix_margin_loss else 0.
    config['constraint_loss_stats'] = {key: stats[key].tolist() for key in [
        'qpos_mean', 'qpos_std', 'action_mean', 'action_std']}
    config['bounded_feedback_residual_output'] = args.bounded_feedback_head
    if args.bounded_feedback_head:
        config['feedback_readout_version'] = 'feedback_tanh_v2'
        config['training_revision'] = 'full_task_trained_smooth_feedback_readout_v2'
        config['experiment_scope'] = 'Complete simulation-only ACT pick carry supported place release retreat; trained bounded absolute-target readout'
    config['real_initial_state'] = [*scene['initial_robot_state']['qpos_rad'],
                                    scene['initial_robot_state']['gripper_total_width_m']]
    if args.previous_gripper_context:
        if args.init_checkpoint is None:raise ValueError('Explicit existing8D or context initializer required')
        previous_config=torch.load(args.init_checkpoint,map_location='cpu',weights_only=False)['config']
        if not previous_config.get('gripper_command_context'):
            from training.command_context import prepare_context_initializer
            context_initializer=output/'artifacts/context_initializer.ckpt'
            config['initialization_legacy_source']=str(args.init_checkpoint.resolve())
            config['init_checkpoint']=str(context_initializer.resolve())
            prepare_context_initializer(args.init_checkpoint,config,context_initializer)
    config_path = output/'config.json'
    config_path.write_text(json.dumps(config, indent=2))
    trainer.NeroActPolicy = FullTaskPolicy
    trainer.fixed_window_samples = partial(complete_windows,command_context=args.previous_gripper_context)
    trainer.fixed_validation_samples = partial(validation_windows,command_context=args.previous_gripper_context)
    if args.previous_gripper_context:
        from training.command_context import ReferenceDataset
        trainer.NeroEpisodicDataset=ReferenceDataset
    trainer.save_checkpoint = save_checkpoint
    torch.set_num_threads(2)
    sys.argv = ['retained_act_trainer', '--config', str(config_path)]
    return trainer.main()


if __name__ == '__main__':
    raise SystemExit(main())
