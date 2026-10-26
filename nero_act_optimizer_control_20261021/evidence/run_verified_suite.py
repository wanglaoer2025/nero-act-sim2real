from pathlib import Path
import sys,subprocess,json,re,hashlib,time
root=Path(sys.argv[1]).resolve()
sources=sorted(p for folder in ['training','executor','simulation','calibration','vision','policy_source','tests'] for p in (root/folder).rglob('*.py'))
def snapshot():return {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
before=snapshot();log=root/'evidence/full_suite.log';start=time.monotonic()
with log.open('w') as out:result=subprocess.run([str(root/'.venv/bin/python'),'-m','pytest','-q'],cwd=root,stdout=out,stderr=subprocess.STDOUT)
text=log.read_text();counts=re.findall(r'(\d+) passed(?:, (\d+) skipped)?',text);after=snapshot()
if result.returncode or not counts or before!=after:raise RuntimeError('suite failed or source changed: '+text[-2000:])
passed,skipped=counts[-1];proof=dict(complete=True,failed=0,tests_passed=int(passed),tests_skipped=int(skipped or 0),source_sha256=before,elapsed_s=time.monotonic()-start)
(root/'evidence/tested_sources.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='source_sha256'}))
