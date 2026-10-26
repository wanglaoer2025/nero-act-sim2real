import hashlib,json,shutil
from pathlib import Path
import h5py,numpy as np
from test_data import episode


def test_mechanism_pilot_accepts_only_qualified_explicit_scope_and_not_formal(tmp_path,monkeypatch):
    import train
    original=train.ROOT;root=tmp_path/'project';root.mkdir()
    for name in ['position_protocol.json','final_position_protocol.json','historical_development_cases.json','observer.py','evaluate.py']:
        shutil.copyfile(original/name,root/name)
    (root/'evidence').mkdir()
    (root/'evidence/implementation_manifest.json').write_text(json.dumps({'sha256':{
        name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['observer.py','evaluate.py']}}))
    monkeypatch.setattr(train,'ROOT',root)
    monkeypatch.setattr(train.torch.cuda,'is_available',lambda:True)
    source=root/'source.h5';base={'pick_xy_m':[.4,0]}
    with h5py.File(source,'w') as f:f.attrs['scene_config']=json.dumps(base)
    protocol=json.loads((root/'position_protocol.json').read_text());rows=[]
    for split,indices,count in [('train',[0,1,3,4,5,6,8,10,11],10),('dev',list(range(6)),5)]:
        for i in indices:
            for repeat in range(count):
                p=root/f'{split}_{i:02d}_{repeat:02d}.h5';episode(p)
                with h5py.File(p,'r+') as f:
                    xy=np.asarray(base['pick_xy_m'])+np.asarray(protocol['positions_mm'][split][i])/1000
                    f.attrs.update(split=split,position_group=f'{split}_{i:02d}',scene_config=json.dumps({'pick_xy_m':xy.tolist()}))
                rows.append(dict(path=str(p),split=split,position_group=f'{split}_{i:02d}',sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(schema='nero_act_transfer_v2_mechanism_pilot_data',complete=True,pilot_only=True,
        source_scene=str(source),episodes=rows,observer_sha256=hashlib.sha256((root/'observer.py').read_bytes()).hexdigest(),
        position_protocol_sha256=hashlib.sha256((root/'position_protocol.json').read_bytes()).hexdigest())
    path=root/'pilot.json';path.write_text(json.dumps(manifest))
    report=train.preflight(path,mode='pilot')
    assert report['allowed'] and report['admission_scope']=='mechanism_pilot_only'
    assert not report['full_main_data_ready'] and report['optimizer_updates']==0
    assert not train.preflight(path,mode='formal')['allowed']
    manifest['pilot_only']=False;path.write_text(json.dumps(manifest))
    assert not train.preflight(path,mode='pilot')['allowed']
    manifest['pilot_only']=True;manifest['episodes']=rows[:-1];path.write_text(json.dumps(manifest))
    assert not train.preflight(path,mode='pilot')['allowed']
