"""Infer the bundled paired recorded observation. Never connects to robot or camera."""
import argparse,json,pickle,hashlib,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'policy_source'))
from src.act.nero_policy import NeroActPolicy
from src.act.nero_dataset import preprocess_images

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'logs/saved_observation');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);bundle=torch.load(ROOT/'model/policy.ckpt',map_location='cpu',weights_only=False);config=dict(bundle['config'],device='cpu',pretrained_backbone=False)
 with (ROOT/'model/dataset_stats.pkl').open('rb') as f:stats=pickle.load(f)
 state=np.array(json.loads((ROOT/'scene/alignment.json').read_text())['current_paired_state'],np.float32);image=np.array(Image.open(ROOT/'scene/real_start.png').convert('RGB'))
 assert config['state_dim']==8 and config['action_dim']==8 and config['chunk_size']==20
 policy=NeroActPolicy(config);policy.load_state_dict(bundle['model'],strict=True);policy.eval()
 with torch.inference_mode():raw=policy(torch.from_numpy((state-stats['qpos_mean'])/stats['qpos_std'])[None],preprocess_images(image[None],config['image_size'])[None])[0].numpy()*stats['action_std']+stats['action_mean']
 assert raw.shape==(20,8) and np.isfinite(raw).all();np.save(args.output/'raw_chunk.npy',raw)
 report={'epoch':bundle['epoch'],'checkpoint_sha256':hashlib.sha256((ROOT/'model/policy.ckpt').read_bytes()).hexdigest(),'raw_chunk_shape':list(raw.shape),'state':state.tolist(),'first_arm_gap_rad':float(np.max(abs(raw[0,:7]-state[:7]))),'first_gripper_target_m':float(raw[0,7]),'raw_first10':raw[:10].tolist(),'motion_sent':False,'live_camera_used':False,'hardware_deployment_allowed':False}
 (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
