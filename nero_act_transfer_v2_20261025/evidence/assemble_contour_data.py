import ast,copy,json,os,sys,time
from pathlib import Path
import h5py
ROOT=Path('/home/user/wq/nero_act_transfer_v2_20261025');sys.path.insert(0,str(ROOT))
from pilot_revalidate import sha
from data_audit import audit_episode
from train import spatial_errors

oldpath=ROOT/'evidence/observer_projected_v1.py'
old=ast.parse(oldpath.read_text());new=ast.parse((ROOT/'observer.py').read_text())
def target(module):
 f=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='current_cup_component')
 return next(n for n in ast.walk(f) if isinstance(n,ast.If) and isinstance(n.test,ast.Call)
             and isinstance(n.test.func,ast.Name) and n.test.func.id=='any')
before,after=target(old),target(new)
assert len(before.body)==1 and isinstance(before.body[0],ast.Raise)
after.body=copy.deepcopy(before.body)
assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)
previous_path=ROOT/'evidence/projected_observer_equivalence.json'
previous=json.loads(previous_path.read_text())
assert previous['compatible'] and previous['accepted_source_episodes']==120
assert previous['new_observer_sha256']==sha(oldpath)
initial=json.loads((ROOT/'data/v2_projected_01/manifest.json').read_text())
extra=json.loads((ROOT/'data/v2_projected_missing_01/manifest.json').read_text())
assert not extra['collection_running'] and len(extra['episodes'])==10
assert extra['observer_sha256']==sha(oldpath)
first=json.loads((ROOT/'data/v2_contour_missing_01/first_full_route.json').read_text())
assert first['accepted'] and first['result']['full_task_verified']
protocol=json.loads((ROOT/'position_protocol.json').read_text())
with h5py.File(initial['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
proof=dict(compatible=True,method='prior-success domain preserved; only previous geometric rejection branch extended with contour fit',
 old_observer_sha256=sha(oldpath),new_observer_sha256=sha(ROOT/'observer.py'),
 inherited120_source_audit=dict(path=str(previous_path),sha256=sha(previous_path)),
 accepted_source_episodes=130,old_data_relabelled=False,truth_used_as_observer_input=False,
 hardware_commands_sent=False,checks=[])
for row in extra['episodes']+[first]:
 assert sha(row['path'])==row['sha256']
 assert audit_episode(row['path'])['main15_ready']
 assert not spatial_errors(row,base,protocol)
 proof['checks'].append(dict(path=row['path'],sha256=row['sha256'],verified=True))
proof_path=ROOT/'evidence/contour_observer_equivalence.json'
assert not proof_path.exists();proof_path.write_text(json.dumps(proof,indent=2)+'\n')
union=ROOT/'data/v2_contour_source_union_01';union.mkdir(exist_ok=False)
source=copy.deepcopy(initial);source.update(episodes=initial['episodes']+extra['episodes'],
 observer_sha256=sha(ROOT/'observer.py'),success_domain_equivalence_proof=str(proof_path),
 previous_failed_attempts_retained_at=str(ROOT/'data/v2_projected_missing_01/manifest.json'))
(union/'manifest.json').write_text(json.dumps(source,indent=2)+'\n')
output=ROOT/'data/v2_contour_01';output.mkdir(exist_ok=False)
manifest=dict(schema=initial['schema'],complete=False,episodes=[],rejected=[],expected_total=150,
 source_scene=initial['source_scene'],source_manifest=str(union/'manifest.json'),previous_version_retained=True,
 observer_sha256=sha(ROOT/'observer.py'),position_protocol_sha256=sha(ROOT/'position_protocol.json'),
 collection_running=False,training_started=False,hardware_commands_sent=False,
 success_domain_equivalence_proof=str(proof_path))
for row in source['episodes']+[first]:
 dest=output/Path(row['path']).name
 os.link(row['path'],dest)
 adopted=copy.deepcopy(row);adopted.update(path=str(dest),adopted_from=row['path'])
 manifest['episodes'].append(adopted)
assert len(manifest['episodes'])==131 and len({r['path'] for r in manifest['episodes']})==131
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(accepted=131,missing=19,observer_sha256=manifest['observer_sha256'])),flush=True)
