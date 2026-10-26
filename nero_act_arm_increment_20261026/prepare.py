"""Zero-update data continuity, causal scales and physical-init equivalence."""
from pathlib import Path
import json
import torch
from torch.utils.data import default_collate
from experiment import BASE,ArmPolicy
from support import reuse_data_proof,fit_delta_statistics,save_json,digest_file,core_state,object_digest
from metrics import representative_dataset,deployment_metrics

ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'nero_act_prefix_supervision_20261025'
MANIFEST=BASE/'data/v2_contour_01/manifest.json'
CHECKPOINT=BASE/'training/formal/policy_20000.ckpt'


def main():
    torch.set_num_threads(2)
    out=ROOT/'preflight';out.mkdir(exist_ok=False)
    proof=reuse_data_proof(PRIOR/'data_verification.json',PRIOR/'source_binding.json',MANIFEST)
    save_json(ROOT/'data_proof.json',proof)
    manifest=json.loads(MANIFEST.read_text())
    paths=[r['path'] for r in manifest['episodes'] if r['split']=='train']
    if len(paths)!=120 or len(manifest['episodes'])!=150:raise RuntimeError('Original120train/30dev required')
    stats=fit_delta_statistics(paths);save_json(ROOT/'delta_stats.json',stats)
    cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if cp['updates']!=20000 or cp['stats']['source_paths']!=paths:raise RuntimeError('Mismatched20Ksource/stats')
    a=ArmPolicy(cp['stats'],stats,variant='A',pretrained=False).cuda().eval();a.load_source(cp)
    b=ArmPolicy(cp['stats'],stats,variant='B',pretrained=False).cuda().eval();b.load_source(cp)
    if object_digest(core_state(a))!=object_digest(cp['model']) or object_digest(core_state(b))!=object_digest(cp['model']):
        raise RuntimeError('Frozen core differs from source')
    dataset,splits=representative_dataset(manifest,cp['stats'])
    errors=[];gripper=[]
    for offset in range(0,len(dataset),32):
        batch=default_collate([dataset[i] for i in range(offset,min(offset+32,len(dataset)))])
        obs=batch['observation'].cuda();image=batch['image'].cuda()
        with torch.inference_mode():
            old=a.source_prediction(obs,image);pa=a(obs,image);pb=b(obs,image)
        for prediction in [pa,pb]:
            errors.append(float(((prediction[:,:,:7]-old[:,:,:7])*a.action_std[:7]).abs().max()))
            gripper.append(float((prediction[:,:,7]-old[:,:,7]).abs().max()))
    if max(errors)>1e-6 or max(gripper)!=0.:raise RuntimeError('Physical initialization is not equivalent')
    metrics=deployment_metrics(a,dataset,splits);save_json(out/'source20K_metrics.json',metrics)
    report=dict(complete=True,optimizer_updates=0,samples=len(dataset),
        source_checkpoint_sha256=digest_file(CHECKPOINT),manifest_sha256=digest_file(MANIFEST),
        frozen_core_sha256=object_digest(cp['model']),maximum_physical_arm_equivalence_error_rad=max(errors),
        maximum_gripper_equivalence_error= max(gripper),train_only_delta_scale_rad=stats['std'],
        old_absolute_scale_rad=cp['stats']['action_std'][:7],
        source_files_sha256={n:digest_file(ROOT/n) for n in ['experiment.py','support.py','prepare.py','metrics.py','delta_stats.json','data_proof.json']},
        hardware_commands_sent=False)
    save_json(out/'completion.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
