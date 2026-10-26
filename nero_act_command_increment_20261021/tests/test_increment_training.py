import pytest
import torch
from tests.test_increment import STATS,observation


def test_target_delta_uses_previous_command_not_measured_width():
    from pipeline import prefix_targets
    actions=torch.zeros(2,20,8);actions[:,0,7]=.0698;actions[:,1,7]=.0696
    q=observation(n=2);q[1,7]=.080;q[1,8]=-1.
    target,delta,valid=prefix_targets(q,actions,torch.zeros(2,20,dtype=torch.bool),STATS)
    torch.testing.assert_close(delta,torch.tensor([[-.2,-.4],[-.2,-.4]]),atol=1e-5,rtol=0)
    assert valid.all() and target.shape==(2,2)


def test_checkpoint_rejects_unfinished_budget_or_changed_bound_source(tmp_path):
    from pipeline import require_checkpoint
    from training.integrity import snapshot
    source=tmp_path/'source.py';source.write_text('fixed')
    checkpoint=tmp_path/'policy_last.ckpt';checkpoint.write_bytes(b'weights')
    proof=dict(complete=True,actual_updates=1000,formal_epochs=0,hardware_deployment_allowed=False,
               source_sha256=snapshot([source]),checkpoint_sha256=snapshot([checkpoint]))
    assert require_checkpoint(proof,checkpoint) is None
    bad=dict(proof,actual_updates=999)
    with pytest.raises(ValueError):require_checkpoint(bad,checkpoint)
    source.write_text('changed')
    with pytest.raises(RuntimeError):require_checkpoint(proof,checkpoint)


def test_frozen_batch_order_rejects_different_samples_and_file_bytes(tmp_path):
    from pipeline import validate_order
    import json,hashlib,numpy as np
    ids=[('episode_1',5,0),('episode_2',6,0)]
    path=tmp_path/'order.npy';np.save(path,np.zeros((1000,32),dtype=np.int64))
    record=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sample_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest())
    assert validate_order(ids,record).shape==(1000,32)
    with pytest.raises(ValueError):validate_order(list(reversed(ids)),record)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):validate_order(ids,record)


def test_approach_reference_requires_original423_and_provenance(tmp_path):
    from pipeline import validated_replay
    from training.integrity import snapshot
    import json,hashlib,numpy as np
    def fixture(n):
        folder=tmp_path/str(n);folder.mkdir();path=folder/'replay.npz'
        np.savez(path,current=np.zeros((n,15)),image=np.zeros((n,1,3,1,1)),target=np.zeros((n,20,8)))
        summary=folder/'summary.json';summary.write_text(json.dumps(dict(optimizer_updates=0,result=dict(simulated_events=[dict(kind='GRASP',physics_step=n)]))))
        trace=folder/'trace.npz';np.savez(trace,query_steps=np.arange(n))
        proof=dict(complete=True,profile='baseline_prefix_v1',zero_update_reference=True,reference_optimizer_updates=0,assessment_role='nominal_development_rehearsal_not_independent_acceptance',source_sha256=snapshot([summary,trace]),replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_summary_path=str(summary),reference_trace_path=str(trace),cutoff_before_GRASP_physics_step=n,samples=n,query_indices=list(range(n)))
        path.with_suffix('.json').write_text(json.dumps(proof));return str(path)
    valid=fixture(423);assert str(tmp_path/'423/replay.npz') in validated_replay(valid)
    with pytest.raises(ValueError):validated_replay(fixture(1))


def test_final_verification_recomputes_failure_instead_of_trusting_flags():
    from run_qualification import verify_outcomes
    from pipeline import PARENT
    import json,copy
    old=json.loads((PARENT/'results/qualification/summary.json').read_text())
    old['records']=json.loads((PARENT/'results/mechanism_original/summary.json').read_text())['records']
    gate=json.loads((PARENT/'results/qualification/input_gate.json').read_text())
    assert verify_outcomes(old,gate) is None
    for change in ['full_task','case','input']:
        bad=copy.deepcopy(old)
        if change=='full_task':bad['full_task_passed']=True
        elif change=='case':bad['records'][0]['passed']=True
        else:bad['exact_and_preservation_gate_passed']=True
        with pytest.raises(ValueError):verify_outcomes(bad,gate)


def test_feature_cache_uses_deployed_single_observation_path():
    from run_training import encode_runtime_features
    from model import IncrementHead,FrozenIncrementPolicy
    from tests.test_increment import TinyAct
    class BatchSensitiveAct(TinyAct):
        def forward(self,q,image):
            return self.model.action_head(torch.ones(len(q),20,128)*len(q)*image.stride(0)/image[0].numel())
    policy=FrozenIncrementPolicy(BatchSensitiveAct(),IncrementHead(128,STATS),STATS,{})
    images=torch.ones(3,1,2,2,3).permute(0,1,4,2,3)
    hidden=encode_runtime_features(policy,observation(n=3),images)
    assert torch.equal(hidden,torch.ones(3,20,128))
