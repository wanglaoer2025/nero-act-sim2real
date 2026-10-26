"""One current-scene ACT reference replay, with owned-process cleanup and RX restoration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'model/frozen_run11'
EVALUATION=ROOT/'evidence/final_frozen_nominal/summary.json'
RECEIPT=ROOT/'evidence/current_replay_authorization.json'
GEOMETRY=ROOT/'evidence/authorized_scene/geometry.json'
OLD_ROOT=ROOT.parent/'nero_act_current_cup_trial_20261019'
OLD_RECORD=OLD_ROOT/'fixed_trial/attempt_01/readonly_restored.json'


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
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(mode='offline_only',hardware_connected=False,motion_sent=False)));return
    from executor.bundle import verify_model_bundle
    from executor.permission import require_experimental_permission
    from executor.evaluation_integrity import verify_frozen_artifacts
    from executor.execution_guards import require_current_scene
    from executor.scene_acceptance import execution_scene
    from executor.replay_trace import verify_replay
    from executor.supervision import install_cleanup_signals,ignore_cleanup_signals
    metadata=verify_model_bundle(MODEL);evaluation=json.loads(EVALUATION.read_text())
    receipt=json.loads(RECEIPT.read_text())
    require_experimental_permission(metadata,evaluation,receipt,now=time.time())
    verify_frozen_artifacts(MODEL/'policy.ckpt',evaluation)
    seal=json.loads((ROOT/'package_sha256.json').read_text())
    for name,expected in seal.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise PermissionError('Sealed replay dependency changed: '+name)
    manifest,_=verify_replay(ROOT,metadata,receipt)
    scene=execution_scene(ROOT,json.loads((MODEL/'scene_config.json').read_text()),manifest,
                          require_evidence=True,receipt=receipt)
    geometry=json.loads(GEOMETRY.read_text())
    require_current_scene(scene,geometry['actual_state'],geometry,now=time.time())
    old=json.loads(OLD_RECORD.read_text());pid=old['pid'];proc=Path('/proc')/str(pid)
    expected=['/usr/bin/python3','-u','-m','executor.driver','--readonly','--output',str(OLD_RECORD)]
    if (old.get('runtime_send_blocked') is not True or old.get('mode')!='receive_only'
            or proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')!=expected
            or proc.joinpath('cwd').resolve()!=OLD_ROOT):
        raise PermissionError('Known receive-only driver identity changed')
    selected={}
    for item in proc.joinpath('environ').read_bytes().split(b'\0'):
        key,sep,value=item.partition(b'=')
        if sep and key.decode() in ['AMENT_PREFIX_PATH','PYTHONPATH','LD_LIBRARY_PATH']:
            selected[key.decode()]=value.decode().replace(str(OLD_ROOT),str(ROOT))
    env=trial_environment(selected);output=ROOT/'evidence/replay_session_01';output.mkdir(exist_ok=False)
    (output/'authorization_receipt.json').write_bytes(RECEIPT.read_bytes())
    (output/'execution_package_sha256.json').write_bytes((ROOT/'package_sha256.json').read_bytes())
    report=dict(mode=manifest['mode'],task_success=False,success=False,processes=[],
                automatic_retry=False,automatic_return=False,source_trace_sha256=manifest['source_rollout_sha256'])
    driver=gate=runtime=None;logs=[];handoff=False
    tx=lambda:int(Path('/sys/class/net/can0/statistics/tx_packets').read_text())
    report['can_tx_before']=tx()
    def spawn(module,params,*,system=False,log_name=None):
        log=None
        if log_name:log=(output/log_name).open('x');logs.append(log)
        cmd=['/usr/bin/python3' if system else str(ROOT/'.venv/bin/python'),'-u','-m',module]+params
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,
            stderr=subprocess.STDOUT if log else None,start_new_session=bool(log),
            stdin=subprocess.DEVNULL if log else None)
        report['processes'].append(dict(pid=child.pid,command=cmd));return child
    def wait_record(path,child):
        end=time.monotonic()+20
        while not path.exists():
            if child.poll() is not None:raise RuntimeError('Owned process exited before startup record')
            if time.monotonic()>end:raise TimeoutError('Owned startup record missing')
            time.sleep(.05)
        return json.loads(path.read_text())
    common=['--model',str(MODEL),'--evaluation',str(EVALUATION),'--receipt',str(RECEIPT)]
    install_cleanup_signals()
    try:
        handoff=True;os.kill(pid,signal.SIGINT);end=time.monotonic()+12
        while proc.exists() and time.monotonic()<end:
            try:
                if not proc.joinpath('cmdline').read_bytes():break
            except FileNotFoundError:break
            time.sleep(.05)
        if proc.exists() and proc.joinpath('cmdline').read_bytes():raise RuntimeError('Known RX driver did not exit')
        runtime=spawn('executor.virtual_bridge',['--synchronize-virtual','--output',str(output/'mock_sync.json')])
        if runtime.wait(timeout=30)!=0:raise RuntimeError('Virtual-only synchronization failed')
        record=output/'driver.json'
        driver=spawn('executor.driver',common+['--execute','--output',str(record)],system=True,log_name='driver.log')
        wait_record(record,driver)
        gate=spawn('executor.gate',common+['--execute','--ros-args',
            '-p','arm_joint_names:=[joint1,joint2,joint3,joint4,joint5,joint6,joint7]',
            '-p','status_topics:=[arm_controller/follow_joint_trajectory/_action/status,gripper_controller/follow_joint_trajectory/_action/status]',
            '-p','arm_sync_tolerance_rad:=0.001','-p','gripper_sync_tolerance_m:=0.0005',
            '-p','state_timeout_s:=0.15'],system=True,log_name='gate.log')
        runtime=spawn('executor.replay',common+['--execute','--geometry',str(GEOMETRY),
            '--output',str(ROOT/'evidence/real_attempt_01')])
        deadline=time.monotonic()+1200
        while runtime.poll() is None:
            if driver.poll() is not None or gate.poll() is not None:
                raise RuntimeError('Driver/protection gate exited during replay')
            if time.monotonic()>deadline:raise TimeoutError('One-shot replay session timed out')
            time.sleep(.05)
        task=json.loads((ROOT/'evidence/real_attempt_01/report.json').read_text())
        report['task_success']=runtime.returncode==0 and task['success'] is True
        report['success']=report['task_success']
    except BaseException as exc:
        report['error']=f'{type(exc).__name__}: {exc}';report['success']=False
    finally:
        ignore_cleanup_signals()
        for name,child in [('runtime',runtime),('gate',gate),('driver',driver)]:
            try:stop_owned(child)
            except Exception as exc:
                report.setdefault('cleanup_errors',[]).append(name+': '+str(exc));report['success']=False
        if handoff:
            try:
                if any(c is not None and c.poll() is None for c in [runtime,gate,driver]):
                    raise RuntimeError('Owned control process still running; cannot create second driver')
                path=output/'readonly_restored.json'
                restored=spawn('executor.driver',['--readonly','--output',str(path)],system=True,log_name='readonly_restored.log')
                record=wait_record(path,restored)
                if (record.get('runtime_send_blocked') is not True or record.get('speed_percent_parameter')!=0
                        or restored.poll() is not None or os.getsid(restored.pid)!=restored.pid):
                    raise RuntimeError('Restored RX driver is not detached/send-blocked')
                report['readonly_restored']=record
            except BaseException as exc:report['restoration_error']=str(exc);report['success']=False
        report['can_tx_after']=tx();(output/'lifecycle.json').write_text(json.dumps(report,indent=2))
        for log in logs:log.close()
        print(json.dumps(report),flush=True)
    if not report['success']:raise SystemExit(1)

if __name__=='__main__':main()
