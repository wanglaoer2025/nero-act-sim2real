"""Read ROS graph, parameters and controller list; never publish actions or change state."""
import json
from pathlib import Path
import time

import rclpy
from rcl_interfaces.srv import GetParameters
from controller_manager_msgs.srv import ListControllers


def main():
    rclpy.init(args=[])
    node = rclpy.create_node('act_fixed_scene_profile_readonly')
    report = dict(mode='ROS_graph_and_GetParameters_only', motion_sent=False,
                  gripper_sent=False, enable_sent=False, parameter_changes=False,
                  sdk_imported=False, wall_s=time.time())
    try:
        end = time.monotonic()+2
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
        report['nodes'] = node.get_node_names_and_namespaces()
        report['control_topics'] = {}
        for topic in ['/feedback/joint_states', '/control/joint_states', '/control/gated_joint_states']:
            report['control_topics'][topic] = dict(
                publishers=[dict(name=i.node_name, namespace=i.node_namespace)
                            for i in node.get_publishers_info_by_topic(topic)],
                subscribers=[dict(name=i.node_name, namespace=i.node_namespace)
                             for i in node.get_subscriptions_info_by_topic(topic)])
        names = ['control_enabled', 'auto_enable', 'speed_percent', 'fast_mode',
                 'pub_rate', 'gripper_default_effort', 'can_port', 'arm_type']
        client = node.create_client(GetParameters, '/agx_arm_ctrl_single_node/get_parameters')
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError('Driver parameter read unavailable')
        future = client.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
        if not future.done():
            raise TimeoutError('Driver parameter read timeout')
        report['driver_parameters'] = {
            name: dict(type=v.type, bool_value=v.bool_value, integer_value=v.integer_value,
                       double_value=v.double_value, string_value=v.string_value)
            for name, v in zip(names, future.result().values)}
        client = node.create_client(ListControllers, '/controller_manager/list_controllers')
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError('Controller inventory unavailable')
        future = client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
        if not future.done():
            raise TimeoutError('Controller inventory timeout')
        report['controllers'] = [dict(name=c.name, state=c.state, type=c.type)
                                 for c in future.result().controller]
        report['read_completed'] = True
    except Exception as exc:
        report['read_completed'] = False
        report['error'] = str(exc)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        path = Path(__file__).with_name('current_ros_profile_readonly.json')
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
