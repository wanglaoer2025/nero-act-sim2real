"""Execute prevalidated ACT phases, checking live feedback; no runtime IK or policy."""
import json
import time
import numpy as np
from executor.ros_execution import session_class
from executor.replay_trace import reference_at,validate_reference
from executor.transport import track_ack
from executor.release_hold import release_reference_transfer


def protect_gripper(rows,contact_width):
    rows=np.asarray(rows,float).copy()
    if contact_width is not None:
        if not .06<=contact_width<=.09:raise ValueError('Accepted measured contact width required')
        # Keep the original12mm contact-gap guard; stop demanding deeper closure at9mm.
        rows[:,7]=np.maximum(rows[:,7],contact_width-.009)
    return rows


def require_phase_start(previous,actual,first):
    previous,actual,first=map(lambda x:np.asarray(x,float),(previous,actual,first))
    if (max(abs(first[:7]-previous[:7]))>.002001 or abs(first[7]-previous[7])>.000501
            or max(abs(first[:7]-actual[:7]))>.03):
        raise RuntimeError('ACT phase start is discontinuous')


def contact_remainder(rows,elapsed,initial,contact_width):
    index=min(len(rows)-2,max(0,int(elapsed/.05)))
    remaining=protect_gripper(rows[index:],contact_width)
    # The first replacement sample interpolates from the still-running command.
    remaining[0,7]=np.clip(remaining[0,7],initial[7]-.0005,initial[7]+.0005)
    validate_reference(remaining)
    return remaining


def join_reference(previous,first):
    previous,first=np.asarray(previous,float),np.asarray(first,float)
    count=max(1,int(np.ceil(max(np.max(abs(first[:7]-previous[:7]))/.002,
                                 abs(first[7]-previous[7])/.0005))))
    return np.array([previous+(first-previous)*i/count for i in range(1,count+1)])


def phase_hold_seconds(name):
    # Operator confirmation expires at180s. Keep both zero-velocity goals active
    # through that pause and the following observation/path validation.
    return 0. if name=='retreat' else 300.


def replay_session_class():
    base=session_class()
    from std_msgs.msg import String
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    class ReplaySession(base):
        def __init__(self,*args,**kwargs):
            self.gate_health=None;self.phase_arm=self.phase_grip=None
            self.contact_floor_width=None
            super().__init__(*args,**kwargs)
            def health(message):
                self.gate_health=dict(json.loads(message.data),received=time.monotonic())
            self.node.create_subscription(String,'/act_gate_health',health,10)

        def enable_monitoring(self):
            deadline=time.monotonic()+2
            while self.gate_health is None:
                if time.monotonic()>deadline:raise RuntimeError('Missing gate health publisher')
                self.spin(.01)
            super().enable_monitoring()

        def guard(self):
            if not self.monitoring or self.cancelling:return
            now=time.monotonic()
            if self.phase_arm is not None:
                a,g=self.phase_arm,self.phase_grip
                arm=reference_at(a['initial'],a['rows'],now-a['start'])
                grip=reference_at(g['initial'],g['rows'],now-g['start'])
                self.last_target=np.r_[arm[:7],grip[7]]
            super().guard()
            status=self.gate_health
            if (status is None or now-status['received']>.15 or status['failed']
                    or not status['receipt_valid']):
                raise RuntimeError('Live ACT gate unhealthy or stale: '+str(status))
            if self.phase_arm is not None:
                end=max(x['start']+len(x['rows'])*.05+x.get('hold_s',0.) for x in [self.phase_arm,self.phase_grip])
                if now-self.phase_arm['start']>.15 and now<end-.02 and not status['driver_open']:
                    raise RuntimeError('Active ACT replay phase has a closed physical command gate')
                if max(abs(np.asarray(self.actual())[:7]-self.last_target[:7]))>.03:
                    raise RuntimeError('Actual arm diverged from the time-aligned ACT replay reference')

        def make_goal(self,rows,group,hold_s=0.):
            columns=list(range(7)) if group==0 else [7]
            goal=FollowJointTrajectory.Goal();goal.trajectory.joint_names=[f'joint{i}' for i in range(1,8)] if group==0 else ['gripper']
            for i,row in enumerate(rows):
                point=JointTrajectoryPoint(positions=[float(row[c]) for c in columns])
                point.time_from_start.sec,point.time_from_start.nanosec=divmod((i+1)*50000000,1000000000)
                goal.trajectory.points.append(point)
            if hold_s:
                point=JointTrajectoryPoint(positions=list(goal.trajectory.points[-1].positions))
                ns=round((len(rows)*.05+hold_s)*1e9)
                point.time_from_start.sec,point.time_from_start.nanosec=divmod(ns,1000000000)
                goal.trajectory.points.append(point)
            return goal

        def submit(self,client,goal,stamp):
            goal.trajectory.header.stamp=stamp
            sent=time.monotonic()
            future=track_ack(client.send_goal_async(goal),goal.trajectory.joint_names,sent,
                             self.scheduled_monotonic-.005,
                             self.report.setdefault('controller_acknowledgements',[]))
            self.pending_goal_futures.append(future)
            return future

        def run_phase(self,name,source_rows,clients,pool,wait_background):
            rows=protect_gripper(source_rows,self.contact_floor_width)
            if name=='approach':
                connection=join_reference(self.last_target,rows[0])
                self.report['start_reference_connection']=dict(
                    source='bounded_connection_from_current_feedback_to_frozen_first_point_not_ACT',rows=connection.tolist())
                rows=np.r_[connection,rows]
            if name=='release':
                # Simulation feedback-settle offsets cannot move a real cup during release.
                measured=np.asarray(self.actual())
                transfer=release_reference_transfer(self.last_target,measured)
                rows[:,:7]=measured[:7]
                self.report['supported_release_arm_hold']=dict(measured_state=measured.tolist(),
                    reference_transfer=transfer.tolist(),source='measured_zero_motion_reference_hold_not_ACT')
                rows=np.r_[transfer,rows]
            if name=='retreat':
                connection=join_reference(self.last_target,rows[0])
                self.report['retreat_reference_connection']=dict(
                    source='bounded_safety_connection_after_operator_clearance_not_ACT',rows=connection.tolist())
                rows=np.r_[connection,rows]
            validate_reference(rows)
            actual=np.asarray(self.actual())
            previous=self.last_target.copy()
            require_phase_start(previous,actual,rows[0])
            if self.held_center is not None:
                safe=len(rows)
                for i,row in enumerate(rows):
                    if self.cup_bottom(row)<-1e-6:safe=i;break
                    if i%16==0:self.spin(.001)
                if safe<len(rows):
                    if name!='carry_place' or safe<2:raise RuntimeError('ACT replay would take held cup through table')
                    self.report.setdefault('cup_floor_truncations',[]).append(dict(phase=name,source_points=len(rows),safe_points=safe))
                    rows=rows[:safe]
            self.validate_rows(rows)
            hold_s=phase_hold_seconds(name)
            goals=wait_background(self,pool.submit(lambda:[self.make_goal(rows,g,hold_s) for g in range(2)]))
            self.guard()
            self.pending_goal_futures=[]
            start=time.monotonic();self.scheduled_monotonic=start+.05
            stamp=self.node.get_clock().now().to_msg()
            self.phase_arm=dict(initial=previous.copy(),rows=rows.copy(),start=start,hold_s=hold_s)
            self.phase_grip=dict(initial=previous.copy(),rows=rows.copy(),start=start,hold_s=hold_s)
            self.awaiting_ack=True
            try:
                futures=[self.submit(c,g,stamp) for c,g in zip(clients,goals)]
                handles=[self.wait(f) for f in futures]
                if not all(h.accepted for h in handles):raise RuntimeError('ACT replay goal rejected')
                self.block_handles=handles
            finally:self.awaiting_ack=False
            results=[h.get_result_async() for h in handles]
            record=dict(name=name,source_rows=np.asarray(source_rows).tolist(),sent_rows=rows.tolist(),started_monotonic_s=start)
            self.report.setdefault('phases',[]).append(record)
            while True:
                self.spin(.003)
                # Protect contact rather than loosening the original12mm force/gap rules.
                if name=='close' and self.contact.state=='holding' and self.contact_floor_width is None:
                    self.contact_floor_width=self.contact.contact_width
                    initial=self.last_target.copy()
                    elapsed=time.monotonic()-self.phase_grip['start']
                    remaining=contact_remainder(rows,elapsed,initial,self.contact_floor_width)
                    # Validate the actual wider-finger path before replacing a live goal.
                    self.validate_rows(remaining)
                    elapsed=time.monotonic()-self.phase_grip['start']
                    initial=self.last_target.copy()
                    remaining=contact_remainder(rows,elapsed,initial,self.contact_floor_width)
                    if len(remaining)>=2:
                        replacement=self.make_goal(remaining,1,hold_s)
                        now=time.monotonic();self.scheduled_monotonic=now+.05
                        self.awaiting_ack=True
                        try:
                            future=self.submit(clients[1],replacement,self.node.get_clock().now().to_msg())
                            handle=self.wait(future)
                            if not handle.accepted:raise RuntimeError('Contact protection replacement rejected')
                        finally:self.awaiting_ack=False
                        self.block_handles=[handles[0],handle]
                        results[1]=handle.get_result_async()
                        self.phase_grip=dict(initial=initial,rows=remaining,start=now,hold_s=hold_s)
                        record['measured_contact_width_m']=self.contact_floor_width
                        record['protected_gripper_remainder']=remaining.tolist()
                for future in results:
                    if future.done():
                        result=future.result()
                        if result.status!=4 or result.result.error_code!=0:
                            raise RuntimeError('A replay controller failed; cancelling arm and gripper')
                end=max(x['start']+len(x['rows'])*.05 for x in [self.phase_arm,self.phase_grip])
                if hold_s and time.monotonic()>=end+.1:
                    if any(f.done() for f in results):raise RuntimeError('Controller hold ended before operator confirmation')
                    record['controller_goals_remain_active_at_unchanged_endpoint']=True
                    break
                if not hold_s and all(f.done() for f in results):break
                if time.monotonic()>end+.25:raise TimeoutError('ACT phase result exceeded original250ms tail')
            self.last_target=np.r_[rows[-1,:7],self.phase_grip['rows'][-1,7]]
            if not hold_s:self.phase_arm=self.phase_grip=None
            self.spin(.1)
            record.update(finished_monotonic_s=time.monotonic(),actual_after=self.actual())
            return rows
    return ReplaySession
