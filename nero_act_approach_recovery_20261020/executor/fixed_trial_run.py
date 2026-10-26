"""Owned-process handoff for the approved single trial; no action without --execute."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from executor.bundle import verify_model_bundle
from executor.fixed_permission import ROOT,MODEL,EVALUATION,require_fixed_trial_permission
from executor.fixed_trial import verify_seal,require_clean_handoff

OLD_ROOT=ROOT.parent/'nero_act_sim_rollout_replay_20261017'
OLD_RECORD=OLD_ROOT/'evidence/readonly_restored_02.json'


def trial_environment(selected):
    env=dict(os.environ);env.update(selected)
    env.update(ROS_DOMAIN_ID='42',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/nero_full_act_mpl')
    return env


def stop_owned(process):
    if process is not None and process.poll() is None:
        try:process.send_signal(signal.SIGINT)
        except ProcessLookupError:return
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(mode='offline_only',hardware_connected=False,motion_sent=False)));return
    receipt=ROOT/'fixed_trial/receipt.json'
    authorization=json.loads(receipt.read_text())
    require_fixed_trial_permission(verify_model_bundle(MODEL),json.loads(EVALUATION.read_text()),
                                  authorization,now=time.time())
    verify_seal()
    attempt=Path(authorization.get('attempt_directory',ROOT/'fixed_trial/attempt_01'))
    old_record=Path(authorization.get('readonly_driver_record',OLD_RECORD))
    old_root=Path(authorization.get('readonly_driver_cwd',OLD_ROOT))
    attempt.mkdir(exist_ok=False)
    (attempt/'authorization_receipt.json').write_bytes(receipt.read_bytes())
    (attempt/'execution_package_sha256.json').write_bytes((ROOT/'package_sha256.json').read_bytes())
    old=json.loads(old_record.read_text());pid=old['pid'];proc=Path('/proc')/str(pid)
    expected=['/usr/bin/python3','-u','-m','executor.driver','--readonly','--output',str(old_record)]
    if (old.get('mode')!='receive_only' or old.get('runtime_send_blocked') is not True
            or proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')!=expected
            or proc.joinpath('cwd').resolve()!=old_root):
        raise PermissionError('Existing read-only driver identity changed; nothing stopped')
    selected={}
    for item in proc.joinpath('environ').read_bytes().split(b'\0'):
        key,sep,value=item.partition(b'=')
        if sep and key.decode() in ['ROS_DOMAIN_ID','AMENT_PREFIX_PATH','PYTHONPATH','LD_LIBRARY_PATH']:
            selected[key.decode()]=value.decode().replace(str(old_root),str(ROOT))
    env=trial_environment(selected)
    report=dict(scope='one_fixed_scene_supervised_trial',success=False,old_readonly_pid=pid,
                automatic_retry=False,automatic_return=False,general14of14_passed=False,
                processes=[],stages=[])
    driver=gate=stage=None;handles=[];handoff_started=False
    py=str(ROOT/'.venv/bin/python')
    common=['--model',str(MODEL),'--evaluation',str(EVALUATION),'--receipt',str(receipt)]
    def spawn(module,params,*,system=False,stage_log=None):
        log=None
        if stage_log:
            log=(attempt/stage_log).open('x');handles.append(log)
        command=['/usr/bin/python3' if system else py,'-u','-m',module]+params
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT if log else None,
                               start_new_session=bool(log))
        report['processes'].append(dict(pid=child.pid,command=command))
        (attempt/'lifecycle.json').write_text(json.dumps(report,indent=2))
        return child
    def wait_record(path,child,timeout=20):
        deadline=time.monotonic()+timeout
        while not path.exists():
            if child.poll() is not None:raise RuntimeError('Owned process exited before its startup record')
            if time.monotonic()>deadline:raise TimeoutError('Owned startup timed out')
            time.sleep(.05)
        return json.loads(path.read_text())
    def start_gate(label):
        params=common+['--execute','--ros-args',
            '-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]',
            '-p','status_topics:=[arm_controller/follow_joint_trajectory/_action/status,gripper_controller/follow_joint_trajectory/_action/status]',
            '-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005',
            '-p','state_timeout_s:=0.15']
        return spawn('executor.fixed_gate',params,stage_log=label+'_gate.log')
    try:
        handoff_started=True;os.kill(pid,signal.SIGINT)
        end=time.monotonic()+12
        while proc.exists() and time.monotonic()<end:time.sleep(.05)
        if proc.exists():raise RuntimeError('Known read-only driver did not exit')
        stage=spawn('executor.virtual_bridge',['--synchronize-virtual','--output',str(attempt/'mock_sync.json')])
        if stage.wait(timeout=30)!=0:raise RuntimeError('Isolated mock bridge synchronization failed')
        record=attempt/'driver.json'
        driver=spawn('executor.fixed_driver',common+['--execute','--output',str(record)],system=True,
                     stage_log='driver.log')
        wait_record(record,driver)
        gate=start_gate('preparation')
        stage=spawn('executor.fixed_trial',['--execute','--stage','prepare','--attempt',str(attempt)])
        code=stage.wait(timeout=150);report['stages'].append(dict(stage='prepare',exit_code=code))
        if code!=0:raise RuntimeError('Preparation failed; no task or gate restart permitted')
        prep=json.loads((attempt/'prepare/report.json').read_text())
        require_clean_handoff(prep)
        if prep.get('success') is not True or prep.get('driver_gate_closed_for_abort') is not True:
            raise RuntimeError('Successful preparation and acknowledged closed gate required')
        if gate.poll() is not None:raise RuntimeError('Preparation gate unexpectedly exited')
        stop_owned(gate);gate=None
        # This is the predeclared normal stage boundary, never an abnormality retry.
        report['normal_closed_gate_handoff']=True
        gate=start_gate('task')
        stage=spawn('executor.fixed_trial',['--execute','--stage','task','--attempt',str(attempt)])
        code=stage.wait(timeout=1100);report['stages'].append(dict(stage='task',exit_code=code))
        report['success']=code==0 and json.loads((attempt/'task/report.json').read_text())['success'] is True
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['success']=False
    finally:
        report['task_success']=report['success']
        for name,process in [('stage',stage),('gate',gate),('driver',driver)]:
            try:stop_owned(process)
            except Exception as exc:
                report.setdefault('cleanup_errors',[]).append(name+': '+str(exc));report['success']=False
        if handoff_started:
            try:
                if any(p is not None and p.poll() is None for p in [stage,gate,driver]):
                    raise RuntimeError('Owned control process has not exited; cannot start another driver')
                record=attempt/'readonly_restored.json'
                restored=spawn('executor.driver',['--readonly','--output',str(record)],system=True,
                               stage_log='readonly_restored.log')
                report['read_only_restored']=wait_record(record,restored)
                if not report['read_only_restored']['runtime_send_blocked']:
                    raise RuntimeError('Read-only restoration send guard missing')
            except Exception as exc:
                report['restoration_error']=str(exc);report['success']=False
        (attempt/'lifecycle.json').write_text(json.dumps(report,indent=2))
        for handle in handles:handle.close()
        print(json.dumps(report,ensure_ascii=False),flush=True)
    if not report['success']:raise SystemExit(1)

if __name__=='__main__':main()
