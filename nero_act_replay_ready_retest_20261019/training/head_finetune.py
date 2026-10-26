"""Fixed-scene fine-tuning of the retained ACT action head; no runtime fallback."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from executor.model import BoundedActPolicy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'policy_source'))
from src.act import train as trainer


class HeadPolicy(BoundedActPolicy):
    def __init__(self, config):
        super().__init__(config)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.model.action_head.parameters():
            parameter.requires_grad_(True)
        self.noise = torch.tensor(config.get('training_observation_noise_normalized', [0.]*8))

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # Preserve frozen batch-normalization buffers as well as weights.
        self.model.action_head.train(mode)
        return self

    def forward(self, qpos, image, actions=None, is_pad=None):
        if actions is not None and self.training:
            qpos = qpos+(torch.rand_like(qpos)*2-1)*self.noise.to(qpos.device)
        prediction = super().forward(qpos, image)
        if actions is None:
            return prediction
        loss = weighted_l1(prediction, actions, is_pad)
        zero = prediction.sum()*0
        return dict(loss=loss, l1=loss, kl=zero, gripper_bce=zero, a_hat=prediction)


def weighted_l1(prediction, target, padding):
    valid = (~padding[:, :20]).to(prediction.dtype)
    weights = valid.clone()
    weights[:, :10] *= 5
    return ((prediction[:, :20]-target[:, :20]).abs()*weights[:, :, None]).sum() / (
        weights.sum()*8).clamp_min(1)


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


def save_checkpoint(path, policy, config, split, epoch, metrics):
    torch.save(dict(model=policy.state_dict(), config=config, split=split, epoch=epoch,
                    metrics=metrics, optimizer_omitted_for_inference=True,
                    hardware_deployment_allowed=False,
                    handoff_scope='complete-task simulation training; hardware unverified'), path)


def main():
    from executor.bundle import verify_model_bundle
    source = ROOT/'model/source_run11'
    metadata = verify_model_bundle(source)
    payload = torch.load(source/'policy.ckpt', map_location='cpu', weights_only=False)
    original = payload['config']
    output = ROOT/'training/current_cup_heads_10ep'
    output.mkdir(exist_ok=False)
    manifest = json.loads(Path(original['manifest']).read_text())
    split = json.loads(Path(original['split_file']).read_text())
    known = {int(Path(r['path']).stem.removeprefix('episode_')) for r in manifest['episodes']}
    added = []
    for name in ['full_manifest.json', 'correction_manifest.json']:
        supplemental = json.loads((ROOT/'training_data_current_cup'/name).read_text())
        if supplemental['rejected']:
            raise ValueError('Rejected data cannot enter fine-tuning')
        for item in supplemental['episodes']:
            episode_id = int(Path(item['path']).stem.removeprefix('episode_'))
            if episode_id in known or item['status'] != 'valid' or item['split'] not in ['train', 'val']:
                raise ValueError('Unique accepted episodes with fixed split required')
            if (item['replay']['passed'] is not True or item['prefix_contract_audit']['passed'] is not True
                    or not (item.get('full_task_verified') or item.get('correction_verified'))):
                raise ValueError('Accepted physics and action-contract evidence required')
            known.add(episode_id)
            split[item['split']].append(episode_id)
            added.append(item)
    if len(added) != 6 or set(split['train']) & set(split['val']):
        raise ValueError('Four complete episodes and two corrections with disjoint splits required')
    if not set(payload['split']['train']).issubset(split['train']):
        raise ValueError('Original training-only statistics provenance must remain in training split')
    (output/'manifest.json').write_text(json.dumps(dict(episodes=manifest['episodes']+added), indent=2))
    (output/'split.json').write_text(json.dumps(split, indent=2))
    config = dict(original)
    config.update(manifest=str(output/'manifest.json'), split_file=str(output/'split.json'),
        init_checkpoint=str(source/'policy.ckpt'), normalization_stats=str(source/'dataset_stats.pkl'),
        checkpoint_dir=str(output/'checkpoints'), artifact_dir=str(output/'artifacts'),
        training_log=str(output/'training.csv'), num_epochs=10, device='cuda', cpu_threads=2,
        lr=.0001, pretrained_backbone=False,
        scene_config=json.loads((ROOT/'training_data_current_cup/scene_config.json').read_text()),
        source_training_scene_config=original['scene_config'],
        training_revision='current_cup_existing_action_head_only',
        trained_parameter_names=['model.action_head.weight', 'model.action_head.bias'],
        source_checkpoint_sha256=metadata['checkpoint_sha256'],
        experimental_scope='Fixed recorded cup scene; full ACT simulation replay candidate; no generalization claim',
        posttraining_static_scene_binding=False)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    trainer.NeroActPolicy = HeadPolicy
    trainer.fixed_window_samples = complete_windows
    trainer.fixed_validation_samples = validation_windows
    trainer.save_checkpoint = save_checkpoint
    torch.set_num_threads(2)
    sys.argv = ['retained_act_trainer', '--config', str(output/'config.json')]
    return trainer.main()


if __name__ == '__main__':
    main()
