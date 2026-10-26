"""SDK-independent asynchronous inference and bounded, time-indexed action prefixes."""
import threading
import time
import numpy as np


class AsyncPolicyWorker:
    def __init__(self,predict):
        self.predict=predict;self.condition=threading.Condition()
        self.pending=None;self.result=None;self.error=None;self.stopping=False
        self.submissions=0;self.replaced_pending=0;self.records=[]
        self.thread=threading.Thread(target=self._run,name='nero_act_inference',daemon=True)
        self.thread.start()

    def submit(self,step,payload):
        with self.condition:
            if self.stopping:raise RuntimeError('Inference worker has stopped')
            if self.error is not None:raise RuntimeError('Inference worker failed') from self.error
            self.replaced_pending+=int(self.pending is not None)
            self.pending=(int(step),payload,time.monotonic());self.submissions+=1
            self.condition.notify_all()

    def latest(self):
        with self.condition:
            if self.error is not None:raise RuntimeError('Inference worker failed') from self.error
            return self.result

    def wait_initial(self,timeout=10):
        return self.wait_for_result(timeout=timeout)

    def wait_for_result(self,minimum_observation_step=None,timeout=10):
        deadline=time.monotonic()+timeout
        with self.condition:
            while (self.result is None or (minimum_observation_step is not None and self.result['observation_step']<minimum_observation_step)) and self.error is None:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError('Requested inference result timed out')
                self.condition.wait(remaining)
        return self.latest()

    def _run(self):
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.stopping:self.condition.wait()
                    if self.stopping:return
                    step,payload,submitted=self.pending;self.pending=None
                started=time.monotonic();chunk,details=self.predict(payload)
                chunk=np.asarray(chunk,dtype=float)
                if chunk.ndim!=2 or chunk.shape[1]!=8 or not np.isfinite(chunk).all():
                    raise ValueError('Inference must return a finite [chunk_steps,8] action array')
                result={'observation_step':step,'chunk':chunk.copy(),'details':details,
                        'submitted_wall_s':submitted,'started_wall_s':started,'completed_wall_s':time.monotonic()}
                with self.condition:
                    self.result=result;self.records.append({k:v for k,v in result.items() if k!='chunk'})
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:self.error=exc;self.condition.notify_all()
        finally:
            if hasattr(self.predict,'close'):self.predict.close()

    def close(self):
        with self.condition:self.stopping=True;self.pending=None;self.condition.notify_all()
        self.thread.join(timeout=10)
        if self.thread.is_alive():raise RuntimeError('Inference worker did not stop')


class ActionPrefix:
    def __init__(self,execute_steps,max_observation_age_steps,max_observation_age_s=None):
        if execute_steps<1 or max_observation_age_steps<execute_steps:
            raise ValueError('Invalid prefix/freshness bounds')
        self.execute_steps=execute_steps;self.max_age=max_observation_age_steps
        if max_observation_age_s is not None and max_observation_age_s<=0:
            raise ValueError('Wall-clock freshness bound must be positive')
        self.max_wall_age=max_observation_age_s
        self.plan=None;self.executed=0;self.last_control_step=None;self.blocks=[]

    def action(self,control_step,latest,now=None):
        if self.last_control_step is not None and control_step<=self.last_control_step:
            raise ValueError('Control step must strictly increase')
        self.last_control_step=control_step
        if latest is not None and (self.plan is None or latest['observation_step']>self.plan['observation_step']):
            self.plan=latest;self.executed=0
            self.blocks.append({'observation_step':latest['observation_step'],'executed_actions':0})
        if self.plan is None:return None,{'hold_reason':'no_prediction'}
        age=control_step-self.plan['observation_step']
        info={'observation_step':self.plan['observation_step'],'action_index':age,'executed_actions_in_block':self.executed}
        if self.max_wall_age is not None:
            observed=self.plan.get('details',{}).get('observation_wall_s')
            if now is None or observed is None:raise ValueError('Wall-clock freshness requires observation and control timestamps')
            wall_age=now-observed;info['observation_age_wall_s']=wall_age
            if wall_age<0 or wall_age>self.max_wall_age:
                return None,{**info,'hold_reason':'stale_wall_observation'}
        if age<0 or age>self.max_age or age>=len(self.plan['chunk']):return None,{**info,'hold_reason':'stale_prediction'}
        if self.executed>=self.execute_steps:return None,{**info,'hold_reason':'prefix_exhausted'}
        action=self.plan['chunk'][age].copy();self.executed+=1
        self.blocks[-1]['executed_actions']=self.executed
        return action,{**info,'executed_actions_in_block':self.executed,'hold_reason':None}


def advance_deadline(previous,now,period):
    """Do not issue catch-up bursts after a missed wall-clock deadline."""
    proposed=previous+period
    return proposed if proposed>now else now+period
