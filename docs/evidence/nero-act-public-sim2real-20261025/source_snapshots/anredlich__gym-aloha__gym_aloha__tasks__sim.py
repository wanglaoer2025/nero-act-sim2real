import collections

import numpy as np
from typing import Any
from dm_control.suite import base
from dm_control.mujoco.engine import Physics
from gym_aloha.utils import get_observation_base
import mujoco

from gym_aloha.constants import (
    START_ARM_POSE,
    START_ARM_POSE_TROSSEN_AI_STATIONARY,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)

BOX_POSE = [None]  # to be changed from outside

"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward

################################TrossenAIStationary######################################

class TrossenAIStationaryTask(base.Task):
    """
    A base task for bimanual manipulation with Trossen AI robotic arms in the Trossen AI Stationary Kit form factor.

    :param random: Random seed for environment variability, defaults to ``None``.
    :param onscreen_render: Whether to enable real-time rendering, defaults to ``False``.
    :param cam_list: List of cameras to capture observations, defaults to ``[]``.
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(random=random)
        self.cam_list = cam_list
        if self.cam_list == []:
            self.cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        """
        Processes the action before passing it to the simulation.

        :param action: The action array containing arm and gripper controls.
        :param physics: The MuJoCo physics simulation instance.
        """
        if action.shape[0]==16:
            left_arm_action = action[:6]
            right_arm_action = action[8 : 8 + 6]
            normalized_left_gripper_action = action[6]
            normalized_right_gripper_action = action[8 + 6]
        elif action.shape[0]==14: #anr to be consistent with BimanualViperXTask above
            left_arm_action = action[:6]
            right_arm_action = action[7 : 7 + 6]
            normalized_left_gripper_action = action[6]
            normalized_right_gripper_action = action[7 + 6]

        # Assign the processed gripper actions
        left_gripper_action = normalized_left_gripper_action
        right_gripper_action = normalized_right_gripper_action

        # Ensure both gripper fingers act oppositely
        full_left_gripper_action = [left_gripper_action, left_gripper_action]
        full_right_gripper_action = [right_gripper_action, right_gripper_action]

        # Concatenate the final action array
        env_action = np.concatenate(
            [
                left_arm_action,
                full_left_gripper_action,
                right_arm_action,
                full_right_gripper_action,
            ]
        )
        super().before_step(env_action, physics)

    def initialize_episode(self, physics: Physics) -> None:
        """
        Sets the state of the environment at the start of each episode.

        :param physics: The MuJoCo physics simulation instance.
        """
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieves the current state of the environment.

        :param physics: The MuJoCo physics simulation instance.
        :return: The environment state.
        """
        env_state = physics.data.qpos.copy()
        return env_state

    def get_position(self, physics: Physics) -> np.ndarray:
        """
        Retrieves the current joint positions.

        :param physics: The MuJoCo physics simulation instance.
        :return: The joint positions.
        """
        position = physics.data.qpos.copy()
        position = np.delete(position, [7, 15]) #anr to be consitent with BimanualViperXTask above
        return position[:14]
        #return position[:16]

    def get_velocity(self, physics: Physics) -> np.ndarray:
        """
        Retrieves the current joint velocities.

        :param physics: The MuJoCo physics simulation instance.
        :return: The joint velocities.
        """
        velocity = physics.data.qvel.copy()
        velocity = np.delete(velocity, [7, 15]) #anr to be consitent with BimanualViperXTask above
        return velocity[:14]
        #return velocity[:16]

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """
        Collects the current observation from the environment.

        :param physics: The MuJoCo physics simulation instance.
        :return: An ordered dictionary containing joint positions, velocities, and environment state.
        """
        obs = get_observation_base(physics, self.cam_list)
        obs["qpos"] = self.get_position(physics)
        obs["qvel"] = self.get_velocity(physics)
        obs["env_state"] = self.get_env_state(physics)
        return obs

    def get_reward(self, physics: Physics) -> int:
        """
        Computes the reward for the current timestep.

        :param physics: The MuJoCo physics simulation instance.
        :raises NotImplementedError: This method must be implemented in subclasses.
        """
        # return whether left gripper is holding the box
        raise NotImplementedError


class TrossenAIStationaryTransferCubeTask(TrossenAIStationaryTask):
    """
    A task where a cube must be transferred between two robotic arms.

    :param random: Random seed for environment variability, defaults to ``None``.
    :param onscreen_render: Whether to enable real-time rendering, defaults to ``False``.
    :param cam_list: List of cameras to capture observations, defaults to ``[]``.
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(
            random=random,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
        self.max_reward = 4
        #self.options: dict[str, Any] | None = None
        self.box_size: list[float] | None = None
        self.box_pos: list[float] | None = None
        self.box_color: list[float] | None = None
        self.arms_pos: list[float] | None = None
        self.arms_ref: list[float] | None = None
        self.tabletop: str | None =None
        self.backdrop: str | None =None
        self.lighting: list | None = None

    def initialize_episode(self, physics: Physics) -> None:
        """
        Initializes the episode, resetting the robot's pose and cube position.

        :param physics: The MuJoCo physics simulation instance.
        """
        # TODO Notice: this function does not randomize the env configuration. Instead, set
        # BOX_POSE from outside reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE_TROSSEN_AI_STATIONARY
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]

        if isinstance(self.box_size,list) and len(self.box_size) == 3:
            physics.named.model.geom_size['red_box'] = self.box_size.copy()
        if isinstance(self.box_pos, list) and len(self.box_pos) == 3:
            if self.box_pos[2]<0.0:
                physics.named.data.qpos['red_box_joint'][2]=-self.box_pos[2] #this only overrides the z component of BOX_POSE!
            else:
                physics.named.data.qpos['red_box_joint'][:3] = self.box_pos.copy() #this overrides BOX_POSE
        if isinstance(self.box_color, list) and len(self.box_color) == 4:
           physics.named.model.geom_rgba['red_box'] = self.box_color.copy()
        left_root_body_id = physics.model.name2id('left/root', 'body')
        right_root_body_id = physics.model.name2id('right/root', 'body')
        if isinstance(self.arms_pos, list) and len(self.arms_pos) == 6:
            physics.model.body_pos[left_root_body_id] = self.arms_pos[:3].copy()
            physics.model.body_pos[right_root_body_id] = self.arms_pos[3:].copy()
            #physics.forward()
        if isinstance(self.arms_ref, list) and len(self.arms_ref) == 12:
            for i in range(6):
                physics.named.model.qpos0[f'left/joint_{i}'] = self.arms_ref[i]
                physics.named.model.qpos0[f'right/joint_{i}'] = self.arms_ref[i+6]
        if isinstance(self.lighting,list) and isinstance(self.lighting[0],list) and len(self.lighting[0]) == 3 and isinstance(self.lighting[1],list) and len(self.lighting[1]) == 3:
            if self.lighting[0][0]>0.0:
                physics.named.model.light_diffuse['top_light_1'][:] = self.lighting[0].copy()
                physics.named.model.light_diffuse['top_light_2'][:] = self.lighting[0].copy()
            else:
                physics.named.model.light_diffuse['top_light_1'][:] = [0.7, 0.7, 0.7]
                physics.named.model.light_diffuse['top_light_2'][:] = [0.7, 0.7, 0.7]
            if self.lighting[1][0]>0.0:
                physics.model.vis.headlight.diffuse[:] = self.lighting[1].copy()
                physics.model.vis.headlight.ambient[:] = self.lighting[1].copy()
            else:
                physics.model.vis.headlight.diffuse[:] = [0.6, 0.65, 0.75]
                physics.model.vis.headlight.ambient[:] = [0.5, 0.5, 0.6]
 
        if isinstance(self.tabletop,str):
            self.switch_tabletop_material(physics=physics,material=self.tabletop)

        if isinstance(self.backdrop,str):
            self.switch_backdrop(physics=physics,backdrop=self.backdrop)

        # if isinstance(self.options, dict): #anr added for task options 
        #     red_box_geom_id = physics.model.name2id('red_box', 'geom')
        #     if 'box_size' in self.options and isinstance(self.options['box_size'], list) and len(self.options['box_size']) == 3:
        #         physics.named.model.geom_size['red_box'] = self.options['box_size'].copy()
        #     if 'box_color' in self.options and isinstance(self.options['box_color'], list) and len(self.options['box_color']) == 4:
        #         physics.named.model.geom_rgba['red_box'] = self.options['box_color'].copy()

        super().initialize_episode(physics)

    def switch_tabletop_material(self, physics: Physics, material: str = 'plain'):
        if not material=='wood'  and not material=='plain'  and not material=='my_desktop':
            return
        #material='wood' or 'plain' or 'my_desktop'              
        try:
            # Access the underlying MuJoCo model
            mj_model = physics.model.ptr            
            # Get material IDs
            wood_material_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, "wood_table")
            plain_material_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, "table")            
            my_desktop_material_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MATERIAL, "my_desktop")
            # Get mesh IDs
            tabletop_mesh_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, "tabletop")
            #tablelegs_mesh_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_MESH, "tablelegs")            
            # Find geoms by their mesh ID
            target_material = wood_material_id if material == 'wood' else my_desktop_material_id if material == 'my_desktop' else plain_material_id            
            for i in range(mj_model.ngeom):
                # Check if this is a mesh geom and matches our target meshes
                if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and mj_model.geom_dataid[i] == tabletop_mesh_id: # or mj_model.geom_dataid[i] == tablelegs_mesh_id)):
                    # Switch the material
                    mj_model.geom_matid[i] = target_material                    
            #print(f"Switched table material to {material}")            
        except Exception as e:
            print(f"Failed to switch material: {e}")

    def switch_backdrop(self, physics: Physics, backdrop: str = 'none'):
        if not backdrop=='my_backdrop' and not backdrop=='none':
            return
        #backdrop='my_backdrop' or 'none'               
        try:
            if backdrop=='my_backdrop':
                physics.named.model.geom_rgba['white_wall'][3] = 0.0
                physics.named.model.geom_rgba['white_wall_plane_bottom'][3] = 1.0
                physics.named.model.geom_rgba['white_wall_plane_top'][3] = 1.0
                physics.named.model.geom_rgba['left_wall_bottom'][3] = 1.0
                physics.named.model.geom_rgba['left_wall_top'][3] = 1.0
                physics.named.model.geom_rgba['right_wall_bottom'][3] = 1.0
                physics.named.model.geom_rgba['right_wall_top'][3] = 1.0
            elif backdrop=='none':
                physics.named.model.geom_rgba['white_wall'][3] = 1.0
                physics.named.model.geom_rgba['white_wall_plane_bottom'][3] = 0.0
                physics.named.model.geom_rgba['white_wall_plane_top'][3] = 0.0
                physics.named.model.geom_rgba['left_wall_bottom'][3] = 0.0
                physics.named.model.geom_rgba['left_wall_top'][3] = 0.0
                physics.named.model.geom_rgba['right_wall_bottom'][3] = 0.0
                physics.named.model.geom_rgba['right_wall_top'][3] = 0.0
        except Exception as e:
            print(f"Failed to switch backdrop: {e}")

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieves the environment state related to the cube position.

        :param physics: The MuJoCo physics simulation instance.
        :return: The environment state.
        """
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics: Physics) -> int:
        """
        Computes the reward based on whether the cube has been transferred successfully.

        :param physics: The MuJoCo physics simulation instance.
        :return: The computed reward which is whether left gripper is holding the box
        """
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = (
            "red_box",
            "left/gripper_follower_left",
        ) in all_contact_pairs
        touch_right_gripper = (
            "red_box",
            "right/gripper_follower_left",
        ) in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        # lifted
        if touch_right_gripper and not touch_table:
            reward = 2
        # attempted transfer
        if touch_left_gripper:
            reward = 3
        # successful transfer
        if touch_left_gripper and not touch_table:
            reward = 4
        return reward
