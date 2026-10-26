"""Use the supplied ACT trainer with padded terminal windows included."""
import sys
sys.path.insert(0,"/home/wq/robot_learning/src")
import hashlib
from pathlib import Path
import h5py
import yaml
import torch
import mujoco
from nero_differentiable_fk import NeroForwardKinematics

ROOT=Path('/home/wq/robot_learning')
sys.path.insert(0,str(ROOT.parent/'sim2real/nero_act_source-offline-ready'))
from src.act import train as trainer
from src.act.nero_policy import NeroActPolicy

_train_windows=trainer.fixed_window_samples
_val_windows=trainer.fixed_validation_samples
execution_query_stride=None
stage_sampling=None


def emphasize_contact_windows(paths,samples,settings):
    """Reweight training windows by recorded teacher stages, never policy inputs."""
    before=int(settings.get('before_close_steps',20))
    close_repeats=int(settings.get('close_repeats',1))
    release_repeats=int(settings.get('release_repeats',1))
    prefix_steps=int(settings.get('correction_prefix_steps',0))
    prefix_repeats=int(settings.get('correction_prefix_repeats',1))
    prefix_parents=[Path(parent).resolve() for parent in settings.get('correction_prefix_parents',[])]
    if before<0 or min(close_repeats,release_repeats)<1:
        raise ValueError('Invalid contact-window sampling settings')
    if prefix_steps<0 or prefix_repeats<1 or (prefix_steps and not prefix_parents):
        raise ValueError('Invalid corrective-prefix sampling settings')
    repeated=[]
    for index,path in enumerate(paths):
        corrective_prefix=prefix_steps>0 and any(Path(path).resolve().is_relative_to(parent) for parent in prefix_parents)
        if corrective_prefix and (index,0) not in samples:
            raise ValueError(f'Corrective initial observation was trimmed from training: {path}')
        with h5py.File(path,'r') as f:
            if 'simulation/stage' not in f:
                raise ValueError(f'Missing recorded teacher stages: {path}')
            stages=f['simulation/stage'].asstr()[:]
        close=[i for i,stage in enumerate(stages) if stage=='close']
        release=[i for i,stage in enumerate(stages) if stage=='release']
        if not close or (settings.get('require_release',True) and not release):raise ValueError(f'Missing required contact stages: {path}')
        for sample in samples:
            if sample[0]!=index:continue
            start=sample[1];weight=1
            if max(0,close[0]-before)<=start<=close[-1]:weight=max(weight,close_repeats)
            if release and max(0,release[0]-before)<=start<=release[-1]:weight=max(weight,release_repeats)
            if corrective_prefix and start<prefix_steps:weight=max(weight,prefix_repeats)
            repeated.extend([sample]*(weight-1))
    return samples+repeated


def near_term_l1(prediction,targets,padding,first_steps,weight,prefix_steps=None):
    weights=(~padding).to(prediction.dtype)
    if prefix_steps is not None:weights[:,prefix_steps:]=0
    weights[:,:first_steps]*=weight
    return ((prediction-targets).abs()*weights[:,:,None]).sum()/(weights.sum()*prediction.shape[-1]).clamp_min(1)


def cache_frozen_backbone(backbone):
    """Cache per-image fixed CNN features; trainable projection/ACT remain uncached."""
    for parameter in backbone.parameters():parameter.requires_grad_(False)
    original=backbone.forward
    cache={}
    position_shared=None
    def forward(images):
        nonlocal position_shared
        keys=[hashlib.sha256(row.detach().cpu().contiguous().numpy().tobytes()).digest() for row in images]
        missing=list(dict.fromkeys(key for key in keys if key not in cache))
        if missing:
            indices=[keys.index(key) for key in missing]
            if position_shared is None and len(indices)==1:indices*=2
            with torch.no_grad():features,positions=original(images[indices])
            if position_shared is None:position_shared=[position.shape[0]==1 for position in positions]
            for index,key in enumerate(missing):
                cache[key]=([feature[index:index+1].detach().clone() for feature in features],
                            [position[:1].detach().clone() if shared else position[index:index+1].detach().clone()
                             for position,shared in zip(positions,position_shared)])
        outputs=[cache[key] for key in keys]
        return ([torch.cat([value[0][i] for value in outputs]) for i in range(len(outputs[0][0]))],
                [outputs[0][1][i] if position_shared[i] else torch.cat([value[1][i] for value in outputs])
                 for i in range(len(outputs[0][1]))])
    backbone.forward=forward
    return cache


class DeploymentMeanPolicy(NeroActPolicy):
    def __init__(self,config):
        super().__init__(config)
        self.first_steps=int(config.get('near_term_steps',10))
        self.first_weight=float(config.get('near_term_weight',10))
        self.prefix_steps=int(config.get('loss_prefix_steps',self.chunk_size))
        if not 1<=self.first_steps<=self.chunk_size or self.first_weight<=0:
            raise ValueError('Invalid near-term loss weights')
        if not self.first_steps<=self.prefix_steps<=self.chunk_size:
            raise ValueError('Loss prefix must cover the weighted near-term steps')
        if config.get('freeze_backbone',False):
            self.feature_caches=[cache_frozen_backbone(backbone) for backbone in self.model.backbones]
        self.reset_initial_joint_head=bool(config.get('reset_initial_joint_head',False))
        if config.get('geometry_projection_only',False):
            if not config.get('task_displacement_features',False) or float(config['weight_decay'])!=0.:
                raise ValueError('Projection-only tuning requires geometry features and zero weight decay to preserve old columns')
            observed=int(config['state_dim'])
            projections={'model.input_proj_robot_state.weight','model.encoder_joint_proj.weight'}
            for name,parameter in self.named_parameters():
                enabled=name in projections
                parameter.requires_grad_(enabled)
                if enabled:
                    if parameter.shape[1]!=observed+6:raise ValueError('Unexpected geometry projection width')
                    parameter.register_hook(lambda gradient,n=observed:torch.cat([torch.zeros_like(gradient[:,:n]),gradient[:,n:]],dim=1))
        self.observation_noise_validation=bool(config.get('observation_noise_during_validation',False))
        bounds=config.get('training_observation_noise_normalized')
        if bounds is not None:
            bounds=torch.tensor(bounds,dtype=torch.float32)
            if bounds.shape!=(int(config['state_dim']),) or not torch.isfinite(bounds).all() or (bounds<0).any():
                raise ValueError('Invalid normalized observation noise bounds')
            self.register_buffer('training_observation_noise_bounds',bounds,persistent=False)
        self.tcp_loss_weight=float(config.get('tcp_position_loss_weight',0.))
        self.tcp_loss_scale=float(config.get('tcp_position_loss_scale_m',.01))
        if self.tcp_loss_weight<0 or self.tcp_loss_scale<=0:
            raise ValueError('Invalid training-only TCP loss settings')
        if self.tcp_loss_weight:
            self.training_fk=NeroForwardKinematics(mujoco.MjModel.from_xml_path(config['tcp_loss_robot_xml']))
            mean=torch.tensor(config['tcp_loss_action_mean'],dtype=torch.float32)
            std=torch.tensor(config['tcp_loss_action_std'],dtype=torch.float32)
            if mean.shape!=(8,) or std.shape!=(8,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
                raise ValueError('TCP loss requires paired eight-dimensional action statistics')
            self.register_buffer('tcp_action_mean',mean,persistent=False)
            self.register_buffer('tcp_action_std',std,persistent=False)

    def load_state_dict(self,state_dict,strict=True):
        result=super().load_state_dict(state_dict,strict=strict)
        if self.reset_initial_joint_head:
            if not self.joint_residual_anchor:raise ValueError('Resetting the joint head requires explicit residual anchoring')
            with torch.no_grad():
                self.model.action_head.weight[:7].zero_();self.model.action_head.bias[:7].zero_()
            self.reset_initial_joint_head=False
        return result

    def forward(self,qpos,image,actions=None,is_pad=None):
        if actions is not None and hasattr(self,'training_observation_noise_bounds') and (self.training or self.observation_noise_validation):
            # Supervised labels remain the true same-cycle commands. Only the
            # observed robot state is corrupted; task coordinates have zero bounds.
            qpos=qpos+(torch.rand_like(qpos)*2-1)*self.training_observation_noise_bounds
        prediction=super().forward(qpos,image)
        if actions is None:return prediction
        loss=near_term_l1(prediction,actions[:,:self.chunk_size],is_pad[:,:self.chunk_size],self.first_steps,self.first_weight,self.prefix_steps)
        objective=loss
        if self.tcp_loss_weight:
            predicted_q=(prediction[:,:self.prefix_steps]*self.tcp_action_std+self.tcp_action_mean)[...,:7]
            target_q=(actions[:,:self.prefix_steps]*self.tcp_action_std+self.tcp_action_mean)[...,:7]
            predicted_tcp=self.training_fk(predicted_q)/self.tcp_loss_scale
            target_tcp=self.training_fk(target_q)/self.tcp_loss_scale
            tcp_loss=near_term_l1(predicted_tcp,target_tcp,is_pad[:,:self.prefix_steps],self.first_steps,self.first_weight)
            objective=loss+self.tcp_loss_weight*tcp_loss
        zero=prediction.sum()*0
        return {'loss':objective,'l1':loss,'kl':zero,'gripper_bce':zero,'a_hat':prediction}


def inference_checkpoint(path,policy,config,split,epoch,metrics):
    torch.save({'model':policy.state_dict(),'config':config,'split':split,'epoch':epoch,
                'metrics':metrics,'optimizer_omitted_for_inference':True,
                'hardware_deployment_allowed':False,
                'handoff_scope':'simulation training; recorded-observation shadow verification only'},path)


def padded_train_windows(paths,chunk_size,**kwargs):
    # The loader already pads future targets; requiring a full chunk excludes
    # exactly the late states where a policy must learn release and retreat.
    samples=_train_windows(paths,1,**kwargs)
    return emphasize_contact_windows(paths,samples,stage_sampling) if stage_sampling else samples


def padded_val_windows(paths,samples_per_episode,chunk_size,*args,**kwargs):
    if execution_query_stride is not None:
        samples=[];records=[]
        for index,path in enumerate(paths):
            with h5py.File(path,'r') as f:length=len(f['action'])
            for start in range(0,length,execution_query_stride):
                samples.append((index,start))
                records.append({'episode_id':int(path.stem.removeprefix('episode_')),'start_ts':start})
        return samples,records
    return _val_windows(paths,samples_per_episode,1,*args,**kwargs)


if __name__=='__main__':
    if '--config' in sys.argv:
        settings=yaml.safe_load(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
        for key in ('checkpoint_dir','artifact_dir','training_log'):
            path=Path(settings[key])
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise FileExistsError(f'{key} already contains output: {path}; choose a new path in the config')
        if settings.get('training_objective')=='deployment_mean_weighted_l1':
            trainer.NeroActPolicy=DeploymentMeanPolicy
        if not settings.get('checkpoint_include_optimizer',True):trainer.save_checkpoint=inference_checkpoint
        execution_query_stride=settings.get('execution_query_stride')
        stage_sampling=settings.get('stage_sampling')
        if execution_query_stride is not None:
            if execution_query_stride!=settings.get('training_stride_steps') or execution_query_stride<=0:
                raise ValueError('Execution stride must be positive and match training stride')
    trainer.fixed_window_samples=padded_train_windows
    trainer.fixed_validation_samples=padded_val_windows
    raise SystemExit(trainer.main())
