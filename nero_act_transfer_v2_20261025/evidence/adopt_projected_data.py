"""Read-only equivalence audit, then immutable hard-link adoption of120 episodes."""
import ast,copy,hashlib,json,os,sys,time
from pathlib import Path
import h5py
ROOT=Path('/home/user/wq/nero_act_transfer_v2_20261025')
sys.path.insert(0,str(ROOT))
from data_audit import audit_episode
from train import spatial_errors

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def dump(node):return ast.dump(node,include_attributes=False)
oldpath=ROOT/'evidence/observer_before_cup_geometry.py';newpath=ROOT/'observer.py'
old=ast.parse(oldpath.read_text());new=ast.parse(newpath.read_text())
oldfunc=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='current_cup_component')
newfunc=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='current_cup_component')
assert dump(oldfunc.body[0])==dump(newfunc.body[0])
ot,nt=oldfunc.body[1],newfunc.body[1]
assert dump(ot.body[0])==dump(nt.body[0])
oe,ne=ot.handlers[0].body,nt.handlers[0].body
assert [dump(x) for x in oe[:-2]]==[dump(x) for x in ne[:len(oe)-2]]
assert dump(oe[-1])==dump(ne[len(oe)-2].body[0])
assert dump(oe[-2].test.left)==dump(ne[len(oe)-2].test.left)
assert dump(oe[-2].test.comparators[0])==dump(ne[len(oe)-2].test.comparators[0])
assert isinstance(oe[-2].test.ops[0],ast.NotEq) and isinstance(ne[len(oe)-2].test.ops[0],ast.Eq)
for module in [old,new]:
 module.body=[n for n in module.body if not isinstance(n,ast.FunctionDef) or n.name!='current_cup_component']
for n in ast.walk(new):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='current_cup_component':
  assert len(n.args)==2 and dump(n.args[1])==dump(ast.parse('self.cylinder',mode='eval').body)
  n.args=n.args[:1]
assert dump(old)==dump(new), 'Other observer semantics changed'
binding=json.loads((ROOT/'evidence/implementation_manifest_before_cup_geometry.json').read_text())
for name,digest in binding['sha256'].items():
 if name!='observer.py':assert sha(ROOT/name)==digest, 'Other bound file changed:'+name
source=ROOT/'data/v2_enclosed_01/manifest.json';original=json.loads(source.read_text())
assert original['observer_sha256']==sha(oldpath) and len(original['episodes'])==120
prior=json.loads((ROOT/'evidence/observer_domain_extension_equivalence.json').read_text())
assert prior['compatible'] and prior['new_observer_sha256']==sha(oldpath)
output=ROOT/'data/v2_projected_01';output.mkdir(exist_ok=False)
proof=dict(compatible=True,method='AST-proven exception-domain-only extension; original successful returns and geometry unchanged',
 old_observer_sha256=sha(oldpath),new_observer_sha256=sha(newpath),source_manifest=str(source),
 source_manifest_sha256=sha(source),previous_equivalence_sha256=sha(ROOT/'evidence/observer_domain_extension_equivalence.json'),
 checks=[],old_data_relabelled=False,truth_used_as_observer_input=False,hardware_commands_sent=False)
with h5py.File(original['source_scene']) as f:base=json.loads(f.attrs['scene_config'])
protocol=json.loads((ROOT/'position_protocol.json').read_text())
manifest=dict(schema=original['schema'],complete=False,episodes=[],rejected=[],expected_total=150,
 source_scene=original['source_scene'],source_manifest=str(source.resolve()),previous_version_retained=True,
 observer_sha256=sha(newpath),position_protocol_sha256=sha(ROOT/'position_protocol.json'),
 collection_running=False,training_started=False,hardware_commands_sent=False,
 success_domain_equivalence_proof=str(ROOT/'evidence/projected_observer_equivalence.json'))
start=time.monotonic()
for row in original['episodes']:
 path=Path(row['path']);digest=sha(path)
 assert digest==row['sha256'], 'Source episode changed:'+str(path)
 audit=audit_episode(path);assert audit['main15_ready'],audit
 assert not spatial_errors(row,base,protocol)
 dest=output/path.name;os.link(path,dest)
 adopted=copy.deepcopy(row);adopted.update(path=str(dest),adopted_from=str(path),byte_identical_source_verified=True)
 manifest['episodes'].append(adopted)
 proof['checks'].append(dict(path=str(path),sha256=digest,frames=audit['frames'],verified=True))
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 if len(proof['checks'])%10==0:print(json.dumps(dict(verified=len(proof['checks']),elapsed_s=time.monotonic()-start)),flush=True)
proof.update(accepted_source_episodes=120,elapsed_s=time.monotonic()-start)
(ROOT/'evidence/projected_observer_equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
print('verified120',flush=True)
