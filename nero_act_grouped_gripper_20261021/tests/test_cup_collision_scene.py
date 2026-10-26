"""Planning-scene transitions with fake ROS messages, never ROS or hardware."""
import copy
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np


def message_modules():
    class CollisionObject:
        ADD, REMOVE = 0, 1
        def __init__(self, id='', operation=0):
            self.id, self.operation = id, operation
            self.header, self.primitives, self.primitive_poses = NS(frame_id=''), [], []
    class PlanningScene:
        def __init__(self, is_diff=False):
            self.is_diff = is_diff
            self.world = NS(collision_objects=[])
            self.robot_state = NS(is_diff=False, attached_collision_objects=[])
            self.allowed_collision_matrix = NS(entry_names=[], entry_values=[],
                                               default_entry_names=[], default_entry_values=[])
    class Pose:
        def __init__(self):
            self.position = NS(x=0., y=0., z=0.)
            self.orientation = NS(x=0., y=0., z=0., w=1.)
    modules = {name: ModuleType(name) for name in ['moveit_msgs', 'moveit_msgs.msg',
        'moveit_msgs.srv', 'geometry_msgs', 'geometry_msgs.msg', 'shape_msgs', 'shape_msgs.msg']}
    messages = modules['moveit_msgs.msg']
    messages.CollisionObject, messages.PlanningScene = CollisionObject, PlanningScene
    messages.AttachedCollisionObject = lambda **kw: NS(**kw)
    messages.AllowedCollisionEntry = lambda enabled: NS(enabled=enabled)
    messages.AllowedCollisionMatrix = lambda **kw: NS(**kw)
    modules['moveit_msgs.srv'].ApplyPlanningScene = NS(Request=lambda **kw: NS(**kw))
    modules['geometry_msgs.msg'].Pose = Pose
    modules['shape_msgs.msg'].SolidPrimitive = type('SolidPrimitive', (), {
        'CYLINDER': 3, '__init__': lambda self, **kw: self.__dict__.update(kw)})
    return modules


class CupCollisionSceneTests(unittest.TestCase):
    def test_mount_exception_requires_a_fixed_base_and_matching_static_table(self):
        from executor import ros_execution
        self.assertTrue(callable(getattr(ros_execution,'require_fixed_table_mount',None)))
        urdf='<robot><joint name="mount" type="fixed"><parent link="world"/><child link="base_link"/></joint></robot>'
        pose=lambda z:NS(position=NS(x=0.,y=0.,z=z),orientation=NS(x=0.,y=0.,z=0.,w=1.))
        table=NS(id='nero_table',header=NS(frame_id='world'),pose=pose(-.02),
            primitives=[NS(type=1,dimensions=[.6,.6,.04])],primitive_poses=[pose(0)])
        ros_execution.require_fixed_table_mount(urdf,table,0.)
        for invalid_urdf in [urdf.replace('fixed','revolute'),urdf.replace('world','link1')]:
            with self.assertRaises(RuntimeError):ros_execution.require_fixed_table_mount(invalid_urdf,table,0.)
        with self.assertRaises(RuntimeError):ros_execution.require_fixed_table_mount(urdf,table,.013)
        table.header.frame_id='link1'
        with self.assertRaises(RuntimeError):ros_execution.require_fixed_table_mount(urdf,table,0.)

    def test_cup_always_present_with_only_finger_contact_exceptions(self):
        from executor import ros_execution
        self.assertTrue(hasattr(ros_execution, 'CupSceneMixin'),
                        'cup transitions must preserve a testable continuous collision representation')
        modules = message_modules()
        with patch.dict(sys.modules, modules):
            class Session(ros_execution.CupSceneMixin):
                def __init__(self):
                    self.scene_config = dict(cup={'height_m': .095, 'outer_diameter_m': .075},
                        pick_xy_m=[-.5, 0], table_top_z_m=0)
                    self.geometry = NS(pose=lambda _: dict(rotation=np.eye(3), position=np.zeros(3)))
                    self.world, self.attached, self.history = {}, {}, []
                    self.state = modules['moveit_msgs.msg'].PlanningScene()
                    self.state.allowed_collision_matrix.entry_names = ['link1', 'link2']
                    self.state.allowed_collision_matrix.entry_values = [NS(enabled=[False, True]),
                                                                           NS(enabled=[True, False])]
                    self.state.allowed_collision_matrix.default_entry_names = ['act_task_cup']
                    self.state.allowed_collision_matrix.default_entry_values = [True]
                    self.held_center = self.held_rotation = None
                def actual(self):return np.zeros(8)
                def scene(self):return copy.deepcopy(self.state)
                def wait(self, future):return future
                def cached_client(self, service, kind):
                    self.assert_service = service
                    return NS(call_async=self.receive)
                def receive(self, request):
                    update = request.scene
                    # Jazzy PlanningScene processes attached updates BEFORE world updates.
                    for item in update.robot_state.attached_collision_objects:
                        obj = item.object
                        if obj.operation == obj.REMOVE:
                            old = self.attached.pop(obj.id, None)
                            if old is not None:self.world[obj.id] = old.object
                        else:
                            self.world.pop(obj.id, None)
                            self.attached[obj.id] = item
                    for obj in update.world.collision_objects:
                        if obj.operation == obj.REMOVE:
                            if obj.id not in self.world:return NS(success=False)
                            self.world.pop(obj.id)
                        else:self.world[obj.id] = obj
                    if update.allowed_collision_matrix.entry_names:
                        self.state.allowed_collision_matrix = update.allowed_collision_matrix
                    self.history.append(copy.deepcopy((self.world, self.attached, self.state.allowed_collision_matrix)))
                    return NS(success=True)
            session = Session()
            session.apply_scene([session.cup_object()], fixed_table_mount=True)
            session.allow_finger_contact(True)
            self.assertIn('act_task_cup', session.world)
            session.attach_actual_source()
            self.assertEqual(set(session.attached['act_task_cup'].touch_links),
                             {'gripper_link1', 'gripper_link2'})
            session.detach_for_supported_release([-.5, -.16])
            self.assertIn('act_task_cup', session.world)
            self.assertAlmostEqual(session.world['act_task_cup'].primitive_poses[0].position.y, -.16)
            session.finish_supported_release([-.5, -.16])
            for index, (world, attached, acm) in enumerate(session.history):
                self.assertEqual(int('act_task_cup' in world)+int('act_task_cup' in attached), 1)
                self.assertTrue(acm.entry_values[0].enabled[1], 'existing SRDF entries must survive')
                self.assertTrue(acm.entry_values[acm.entry_names.index('base_link')].enabled[
                    acm.entry_names.index('nero_table')], 'only fixed mount contact is exempted')
                self.assertFalse(acm.entry_values[acm.entry_names.index('link1')].enabled[
                    acm.entry_names.index('nero_table')], 'moving-link/table checks must remain')
                if index == 0:continue
                self.assertEqual(acm.default_entry_values, [False])
                cup = acm.entry_names.index('act_task_cup')
                for link in ['link1', 'link2']:
                    self.assertFalse(acm.entry_values[cup].enabled[acm.entry_names.index(link)])
                for link in ['gripper_link1', 'gripper_link2']:
                    finger = acm.entry_names.index(link)
                    self.assertEqual(acm.entry_values[cup].enabled[finger], index < len(session.history)-1)
                    self.assertEqual(acm.entry_values[finger].enabled[cup], index < len(session.history)-1)


if __name__ == '__main__':unittest.main()
