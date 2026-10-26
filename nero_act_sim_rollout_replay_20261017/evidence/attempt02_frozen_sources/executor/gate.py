"""Task-specific low-force ROS gate. Default validates permission without creating a publisher."""
import argparse
import json
import math
from pathlib import Path
import runpy
import time

from executor.contact import contact_sync_allowed
from executor.permission import require_experimental_permission
from executor.bundle import verify_model_bundle

GATE_SOURCE = Path('/home/user/projects/nero_ros2_ws/src/agx_arm_ros/src/agx_arm_moveit/scripts/agx_arm_control_gate')


class GateLatch:
    def __init__(self):
        self.open, self.failed, self.settled = False, False, False
        self.pending, self.sequence = None, 0

    def stop(self):
        self.failed = True
        self.open = False
        self.settled = False
        # Invalidate an enable already in flight before requesting the final close.
        self.pending = None

    def request(self, desired):
        if desired and self.failed:
            return None
        if desired and (not self.settled or self.pending is not None and not self.pending[1]):
            return None
        if self.pending is not None and self.pending[1] == desired:
            return None
        if self.pending is None and self.settled and self.open == desired:
            return None
        self.sequence += 1
        self.pending = (self.sequence, bool(desired))
        if not desired:
            self.open = False
        return self.sequence

    def acknowledge(self, sequence, success):
        if self.pending is None or self.pending[0] != sequence:
            return
        self.open = bool(success and self.pending[1])
        self.failed |= not success
        self.settled = True
        self.pending = None


def gate_class():
    base = runpy.run_path(str(GATE_SOURCE))['ControlGate']
    from std_srvs.srv import SetBool, Trigger
    from std_msgs.msg import Empty,String

    class ContactGate(base):
        def __init__(self, *, ttl_s):
            self.latch = GateLatch()
            self.contact_permission = False
            self.heartbeat = self.feedback_advance = 0.
            self.feedback_stamp = None
            self.feedback_force = float('nan')
            self.receipt_deadline = time.monotonic()+ttl_s
            self.ever_opened = False
            self.failure_reason = None
            self.last_health_publish = 0.
            super().__init__()
            self.health_publisher = self.create_publisher(String,'/act_gate_health',10)
            if (self.arm_joint_names != [f'joint{i}' for i in range(1, 8)]
                    or self.arm_sync_tolerance_rad != .001 or self.gripper_sync_tolerance_m != .0005
                    or self.state_timeout_s != .15 or len(self.status_topics) != 2):
                raise ValueError('Explicit seven-axis strict gate parameters required')
            self.create_subscription(Empty, '/act_full_heartbeat', self._heartbeat, 10)
            self.create_service(SetBool, '/act_contact_permission', self._contact_permission)
            self.create_service(Trigger, '/act_task_stop', self._stop_task)
            self.watchdog = self.create_timer(.02, self._update_gate)

        def _heartbeat(self, _message):
            self.heartbeat = time.monotonic()

        def _stop_task(self, _request, response):
            self.latch.stop()
            self.failure_reason = self.failure_reason or 'explicit_terminal_stop'
            self._set_gate(False)
            response.success = True
            response.message = 'Stop latched permanently for this attempt'
            return response

        def _publish_health(self):
            now=time.monotonic()
            if now-self.last_health_publish<.02:return
            self.last_health_publish=now
            self.health_publisher.publish(String(data=json.dumps(dict(failed=self.latch.failed,
                driver_open=self.latch.open,reason=self.failure_reason,
                receipt_valid=now<self.receipt_deadline))))

        def _contact_permission(self, request, response):
            self.contact_permission = bool(request.data)
            response.success = True
            return response

        def _feedback_callback(self, msg):
            stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            now = time.monotonic()
            if self.feedback_stamp is None or stamp > self.feedback_stamp:
                self.feedback_advance = now
            elif stamp < self.feedback_stamp:
                self.latch.failed = True
            self.feedback_stamp = stamp
            self.feedback_force = float('nan')
            if len(msg.effort) == len(msg.name) and 'gripper' in msg.name:
                self.feedback_force = msg.effort[list(msg.name).index('gripper')]
            super()._feedback_callback(msg)

        def _controller_synchronized(self, topic_name):
            if super()._controller_synchronized(topic_name):
                return True
            if 'gripper_controller' not in topic_name:
                return False
            return contact_sync_allowed(command=self._control.get('gripper', float('nan')),
                feedback=self._feedback.get('gripper', float('nan')), force=self.feedback_force,
                age_s=time.monotonic()-self._feedback_time, authorized=self.contact_permission)

        def _update_gate(self):
            now = time.monotonic()
            ros_age = (self.get_clock().now().nanoseconds*1e-9-self.feedback_stamp
                       if self.feedback_stamp is not None else math.inf)
            names = self.arm_joint_names+['gripper']
            finite = (all(name in self._feedback and name in self._control
                          and math.isfinite(self._feedback[name]) and math.isfinite(self._control[name])
                          for name in names) and math.isfinite(self.feedback_force))
            safe = (finite and not self.latch.failed and now < self.receipt_deadline
                    and now-self.heartbeat <= .15 and now-self.feedback_advance <= .15
                    and now-self._feedback_time <= .15 and now-self._control_time <= .15
                    and -.05 <= ros_age <= .25 and abs(self.feedback_force) <= 1.25)
            if safe:
                safe = (max(abs(self._feedback[n]-self._control[n]) for n in self.arm_joint_names) <= .03
                        and abs(self._feedback['gripper']-self._control['gripper']) <= .012)
            if not safe:
                if self.ever_opened:
                    self.latch.failed = True
                    self.failure_reason=self.failure_reason or json.dumps(dict(
                        finite=finite,heartbeat_age=now-self.heartbeat,
                        feedback_age=now-self._feedback_time,control_age=now-self._control_time,
                        feedback_advance_age=now-self.feedback_advance,ros_age=ros_age,
                        force=self.feedback_force,receipt_valid=now<self.receipt_deadline))
                self._set_gate(False)
                self._publish_health()
                return
            super()._update_gate()
            self._publish_health()

        def _control_callback(self, msg):
            # ACT predicts width only; every forwarded gripper target uses the declared1N cap.
            msg.effort = [1. if name == 'gripper' else 0. for name in msg.name]
            super()._control_callback(msg)

        def _set_gate(self, desired):
            if not self._gate_client.wait_for_service(timeout_sec=0.):
                self._gate_open = False
                self.latch.open = False
                self.latch.settled = False
                return
            sequence = self.latch.request(desired)
            self._gate_open = self.latch.open
            if sequence is None:
                return
            future = self._gate_client.call_async(SetBool.Request(data=bool(desired)))
            def acknowledged(reply):
                success = False
                try:success = bool(reply.result().success)
                except Exception:self.latch.failed = True
                stale = self.latch.pending is None or self.latch.pending[0] != sequence
                self.latch.acknowledge(sequence, success)
                self._gate_open = self.latch.open
                self.ever_opened |= self._gate_open
                if stale and desired and not self._gate_open:
                    if self.latch.pending is None:
                        self.latch.settled = False
                    self._set_gate(False)
            future.add_done_callback(acknowledged)
    return ContactGate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'evaluation', 'receipt']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args, ros_args = parser.parse_known_args()
    metadata = verify_model_bundle(args.model)
    receipt = json.loads(args.receipt.read_text())
    require_experimental_permission(metadata, json.loads(args.evaluation.read_text()), receipt, now=time.time())
    if not args.execute:
        print(json.dumps(dict(mode='permission_check_only',publisher_created=False)))
        return
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    # Keep the ROS context alive until the final close has been acknowledged.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = gate_class()(ttl_s=receipt['authorized_wall_s']+1800-time.time())
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.latch.stop()
            node._set_gate(False)
            deadline = time.monotonic()+.5
            while node.latch.pending is not None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.01)
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':main()
