"""Verify immutable package files without loading models or accessing hardware."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
entries=json.loads((ROOT/'package_sha256.json').read_text())
for name,digest in entries.items():
 path=(ROOT/name).resolve()
 if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('checksum mismatch: '+name)
result=json.loads((ROOT/'verification.json').read_text());result['verified_files']=len(entries);print(json.dumps(result,indent=2))
