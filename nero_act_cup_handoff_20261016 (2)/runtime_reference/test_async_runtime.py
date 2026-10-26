import sys
import threading
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from async_runtime import AsyncPolicyWorker,ActionPrefix,advance_deadline


class RuntimeTests(unittest.TestCase):
    def test_new_prediction_replaces_old_close_without_temporal_blending(self):
        prefix=ActionPrefix(10,15)
        old=np.zeros((100,8));new=np.zeros((100,8));new[:,7]=.095
        old_plan={'observation_step':0,'chunk':old}
        new_plan={'observation_step':6,'chunk':new}
        self.assertEqual(prefix.action(7,old_plan)[0][7],0.)
        self.assertEqual(prefix.action(8,new_plan)[0][7],.095)
        # Even an out-of-order older result must not restore its closure.
        self.assertEqual(prefix.action(9,old_plan)[0][7],.095)

    def test_warmup_result_is_not_mistaken_for_requested_fresh_initial_observation(self):
        def predict(value):return np.full((10,8),value),{}
        worker=AsyncPolicyWorker(predict)
        try:
            worker.submit(-1,-1);self.assertEqual(worker.wait_initial()['observation_step'],-1)
            worker.submit(0,0)
            result=worker.wait_for_result(minimum_observation_step=0,timeout=1)
            self.assertEqual(result['observation_step'],0)
            self.assertTrue(np.all(result['chunk']==0))
        finally:worker.close()

    def test_delayed_plan_skips_elapsed_actions_and_exhausts_bounded_prefix(self):
        prefix=ActionPrefix(3,5)
        chunk=np.tile(np.arange(8),(10,1)).astype(float);chunk[:,0]=np.arange(10)
        plan={'observation_step':0,'chunk':chunk}
        action,info=prefix.action(2,plan)
        self.assertEqual(action[0],2)
        self.assertEqual(info['action_index'],2)
        prefix.action(3,plan);prefix.action(4,plan)
        action,info=prefix.action(5,plan)
        self.assertIsNone(action);self.assertEqual(info['hold_reason'],'prefix_exhausted')
        action,info=prefix.action(6,plan)
        self.assertIsNone(action);self.assertEqual(info['hold_reason'],'stale_prediction')

    def test_busy_worker_keeps_only_latest_pending_observation(self):
        started=threading.Event();release=threading.Event();third=threading.Event();seen=[]
        def predict(payload):
            seen.append(payload)
            if payload==0:started.set();release.wait(2)
            if payload==2:third.set()
            return np.full((10,8),payload),{}
        worker=AsyncPolicyWorker(predict)
        try:
            worker.submit(0,0);self.assertTrue(started.wait(1))
            worker.submit(1,1);worker.submit(2,2);release.set()
            self.assertTrue(third.wait(1));worker.wait_initial()
            worker.close()
            self.assertEqual(seen,[0,2]);self.assertEqual(worker.replaced_pending,1)
            self.assertEqual(worker.latest()['observation_step'],2)
        finally:release.set();worker.close()

    def test_deadline_does_not_burst_after_overrun(self):
        self.assertAlmostEqual(advance_deadline(1,1.01,.05),1.05)
        self.assertAlmostEqual(advance_deadline(1,1.2,.05),1.25)

    def test_wall_clock_staleness_holds_even_when_step_age_is_small(self):
        prefix=ActionPrefix(10,15,.75)
        plan={'observation_step':0,'chunk':np.zeros((100,8)),
              'details':{'observation_wall_s':10.0}}
        action,info=prefix.action(0,plan,10.2)
        self.assertIsNotNone(action)
        action,info=prefix.action(1,plan,10.8)
        self.assertIsNone(action)
        self.assertEqual(info['hold_reason'],'stale_wall_observation')


if __name__=='__main__':unittest.main()
