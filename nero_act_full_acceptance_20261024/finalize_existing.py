"""Finalize persisted seven-case evidence after the legacy audit-path error."""
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run_01'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def finalize_summary(raw,expected_ids):
    ids=[row['id'] for row in raw['records']]
    if ids!=expected_ids or len(set(ids))!=len(ids):
        raise ValueError('Every expected case must be saved exactly once')
    result=copy.deepcopy(raw)
    result.update(complete=True,finalized=True,total=len(ids),
        passed=sum(row['full_task_passed'] for row in raw['records']),
        full_suite_passed=all(row['full_task_passed'] for row in raw['records']),
        recovered_from_postprocessing_failure=True,underlying_rollout_command_exit_success=False,
        in_memory_policy_comparison_completed=False,
        in_memory_policy_comparison_note='Legacy finally raised before the planned final in-memory comparison; no such comparison is claimed.',
        postprocessing_error='Legacy dense audit looked for contact_guard.py in the output directory; actual guard source is already bound in source_sha256.json.')
    return result


def main():
    from run import sealed
    raw=json.loads((RUN/'summary.json').read_text())
    protocol=json.loads((RUN/'protocol.json').read_text())
    sources=json.loads((RUN/'source_sha256.json').read_text())
    for path,expected in sources.items():
        if digest(path)!=expected:raise ValueError('Source changed: '+path)
    for row in raw['records']:
        persisted=json.loads((RUN/str(row['id'])/'result.json').read_text())
        if persisted!=row['result']:raise ValueError('Saved result mismatch')
        if sealed.evaluate_dense.evaluation.full_task_passed(persisted)!=row['full_task_passed']:
            raise ValueError('Original completion predicate mismatch')
    result=finalize_summary(raw,protocol['expected_case_ids'])
    result.update(all_recorded_sources_unchanged=True,protected_source_count=len(sources),
        original_checkpoint_file_unchanged=True,
        raw_incomplete_summary_sha256=digest(RUN/'summary.json'),
        original_entry_sha256=digest(ROOT/'run.py'),
        finalization_entry_sha256=digest(Path(__file__)))
    (RUN/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
