"""Latest gate telemetry, received independently of blocking SDK reads."""
import json
import math
import threading
import time


def require_gate_health(status, now):
    try:
        healthy = (status['failed'] is False and status['receipt_valid'] is True
                   and type(status['driver_open']) is bool
                   and all(math.isfinite(status[key]) and 0 <= now-status[key] <= .15
                           for key in ('received', 'published_monotonic_s')))
    except (TypeError, KeyError, ValueError):
        healthy = False
    if not healthy:
        raise RuntimeError('Live ACT gate unhealthy or stale: '+str(status))


class GateHealthReceiver:
    def __init__(self, context):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        self.latest = None
        self.node = rclpy.create_node('act_replay_gate_health_receiver', context=context)
        self.executor = SingleThreadedExecutor(context=context)
        self.node.create_subscription(String, '/act_gate_health', self.receive,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.run, name='act_gate_health', daemon=True)
        self.thread.start()

    def receive(self, message):
        try:
            self.latest = dict(json.loads(message.data), received=time.monotonic())
        except (TypeError, ValueError) as exc:
            self.latest = dict(failed=True, reason='Malformed gate telemetry: '+str(exc))

    def run(self):
        try:
            self.executor.spin()
        except Exception as exc:
            self.latest = dict(failed=True, reason='Gate telemetry receiver stopped: '+str(exc))

    def close(self):
        self.executor.shutdown(timeout_sec=2.)
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError('Gate telemetry receiver did not stop')
        self.node.destroy_node()
