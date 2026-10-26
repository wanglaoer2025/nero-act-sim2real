import pytest


def test_preflight_does_not_allow_missing_data_or_cpu_formal_training(tmp_path):
    from train import preflight
    report=preflight(tmp_path/'missing.json')
    assert not report['allowed'] and 'qualified_data_manifest_missing' in report['blockers']
    assert report['optimizer_updates']==0


def test_training_cli_defaults_to_check_only(tmp_path):
    from train import parse_arguments
    args=parse_arguments(['--manifest',str(tmp_path/'missing.json'),'--output',str(tmp_path/'report')])
    assert args.mode=='check'
    with pytest.raises(SystemExit):parse_arguments(['--mode','formal','--updates','100000'])


def test_unreadable_manifest_is_a_reported_blocker_not_an_optimization(tmp_path):
    from train import preflight
    p=tmp_path/'broken.json';p.write_text('{')
    report=preflight(p)
    assert not report['allowed'] and 'qualified_manifest_unreadable' in report['blockers']
