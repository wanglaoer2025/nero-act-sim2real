# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities to control a robot.

Useful to record a dataset, replay a recorded episode, run the policy on your robot
and record an evaluation dataset, and to recalibrate your robot if needed.

Examples of usage:

- Recalibrate your robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=calibrate
```

- Unlimited teleoperation at highest frequency (~200 Hz is expected), to exit with CTRL+C:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --robot.cameras='{}' \
    --control.type=teleoperate

# Add the cameras from the robot definition to visualize them:
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate
```

- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=teleoperate \
    --control.fps=30
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/koch_test \
    --control.num_episodes=1 \
    --control.push_to_hub=True
```

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/koch_test \
    --episode-index 0
```

- Replay this test episode:
```bash
python lerobot/scripts/control_robot.py replay \
    --robot.type=so100 \
    --control.type=replay \
    --control.fps=30 \
    --control.repo_id=$USER/koch_test \
    --control.episode=0
```

- Record a full dataset in order to train a policy, with 2 seconds of warmup,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_robot.py record \
    --robot.type=so100 \
    --control.type=record \
    --control.fps 30 \
    --control.repo_id=$USER/koch_pick_place_lego \
    --control.num_episodes=50 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

- For remote controlled robots like LeKiwi, run this script on the robot edge device (e.g. RaspBerryPi):
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=lekiwi \
  --control.type=remote_robot
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resseting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command and adding `--control.resume=true`.

- Train on this dataset with the ACT policy:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --device=cuda \
  --wandb.enable=true
```

- Run the pretrained policy on the robot:
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so100 \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="Grasp a lego block and put it in the bin." \
    --control.repo_id=$USER/eval_act_koch_pick_place_lego \
    --control.num_episodes=10 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10 \
    --control.push_to_hub=true \
    --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/080000/pretrained_model
```
"""

import logging
import time
from dataclasses import asdict
from pprint import pformat
import cv2
import random

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy, get_policy_class
from lerobot.common.policies.scripted_policy import PickAndTransferPolicy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    #record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.robot_devices.control_utils import predict_action, is_headless
from lerobot.common.datasets.image_writer import safe_stop_image_writer #anr added
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.common.utils.utils import get_safe_torch_device #anr added
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs import parser
import gym_aloha
from gym_aloha.tasks.sim import BOX_POSE
import gymnasium as gym
import dm_env
import torch
import numpy

#######################################################################################
# Control utilities
########################################################################################

@safe_stop_image_writer
def record_sim_episode(
    robot,
    dataset: LeRobotDataset | None = None,
    env=None,
    events=None,
    episode_time_s=None,
    display_cameras=False,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    episode = None,
):
    # TODO(rcadene): Add option to record logs
    #if not robot.is_connected:
    #    robot.connect()

    device = "cuda" #anr get this from policy or policy config
    if policy is not None:
        policy.reset()

    if events is None:
        events = {"exit_early": False}

    if episode_time_s is None:
        episode_time_s = float("inf")

    #if teleoperate and policy is not None:
    #    raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    if policy is None:
        ecount=0
         # make sure the sim_env has the same object configurations as ee_sim_env
        BOX_POSE[0] = episode[ecount].observation["env_state"].copy()
        ecount+=1
        numpy_observation, info = env.reset(seed=None,options='do_not_reset_BOX_POSE')
    else:
        numpy_observation, info = env.reset()
    if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
        ts=dm_env.TimeStep(step_type=dm_env.StepType.FIRST,reward=None,discount=None,observation=info['raw_obs'])
        print(f"cube position={ts.observation['env_state'][:3]}")
        episode.append(ts)

    #timestamp = 0
    #start_episode_t = time.perf_counter()
    #while timestamp < episode_time_s:
    for step in range(env._max_episode_steps):
        #start_loop_t = time.perf_counter()

        #if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
        #    numpy_observation["agent_pos"] = numpy.delete(numpy_observation["agent_pos"], [7, 15])

        #convert from gym-aloha convention:
        state = torch.from_numpy(numpy_observation["agent_pos"]).to(torch.float32)
        image = [] #anr added
        cam_list = numpy_observation["pixels"].keys()
        for cam in cam_list:
            image.append(torch.from_numpy(numpy_observation["pixels"][cam])) #anr new

        # Create the policy input dictionary
        data_observation = {
            "observation.state": state,
             **{f"observation.images.{cam}": image[i] for i, cam in enumerate(cam_list)}
        }

        if display_cameras and not is_headless():
            image_keys = [key for key in data_observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(data_observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if policy is not None:
            #print(f'{ts.observation["env_state"][:3]} step={step}')
            if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
                policy.box_size=env.unwrapped._env.task.box_size.copy()
                action = policy(ts)
            else:
                #action = policy.select_action(observation)
                action = predict_action(
                data_observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )

            # Prepare the action for the environment
            if isinstance(action, torch.Tensor):
                numpy_action = action.squeeze(0).to("cpu").numpy()
            else:
                numpy_action = action
        else:
            numpy_action=episode[ecount].observation["qpos"].astype(numpy.float32)
            if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
                numpy_action = numpy.delete(numpy_action, [7, 15]) #anr this stays because ee qpos is size 16
            ecount+=1

        if dataset is not None and env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
            action = {"action": numpy_action}
            frame = {**data_observation, **action, "task": single_task}
            dataset.add_frame(frame)

        #if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
        #    temp_array = numpy.insert(numpy_action, 7, numpy_action[6])
        #    numpy_action = numpy.insert(temp_array, 15, numpy_action[13])

        #if teleoperate:
        #    observation, action = robot.teleop_step(record_data=True)
        #else:
            #observation = robot.capture_observation()
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
            if terminated:
                ts=dm_env.TimeStep(dm_env.StepType.LAST, reward, 1.0, observation=info['raw_obs'])
            else:
                ts=dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation=info['raw_obs'])
            episode.append(ts)

        if not env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':    
            if terminated:
                print(f'nsteps={step}')
                return

    if policy is None and step>=env._max_episode_steps-1:
        flag=1

        #if fps is not None:
        #    dt_s = time.perf_counter() - start_loop_t
        #    busy_wait(1 / fps - dt_s)

        #dt_s = time.perf_counter() - start_loop_t
        #log_control_info(robot, dt_s, fps=fps)

        #timestamp = time.perf_counter() - start_episode_t
        #if events["exit_early"]:
        #    events["exit_early"] = False
        #    break

        # if policy is not None:
        #     policy.reset()

def random_env_options(env,env2,rand_ops):
    set1={'box_size':[[0.02,0.02,0.02],[0.0125,0.0125,0.0125]],'box_color':[[1,0,0,1],[0,1,0,1],[0,0,1,1]],
          'arms_pos':[[-0.4575, 0.0, 0.02, 0.4575, 0.0, 0.02],[-0.4575, -0.019, 0.02, 0.4575, -0.019, 0.02],[-0.4575, 0.019, 0.02, 0.4575, 0.019, 0.02]],
          'arms_ref':[[0,-0.015,0.015,0,0,0,0,-0.025,0.025,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0],[0,0.015,-0.015,0,0,0,0,0.025,-0.025,0,0,0]],
          'lighting':[[[0.3,0.3,0.3],[0.3,0.3,0.3]],[[0.1,0.1,0.1],[0.5,0.5,0.5]],[[-1,0,0],[-1,0,0]]]} #-1 -> default lighting
    set2={'lighting':[[[0.3,0.3,0.3],[0.3,0.3,0.3]],[[0.1,0.1,0.1],[-0.5,0.5,0.5]]],
          'tabletop':['my_desktop','wood']}
    op_sets={'set1':set1,'set2':set2}
    for op in rand_ops:
        if '-' in op:
            option, set = op.split('-', 1)  # maxsplit=1 in case of multiple '-'
        else:
            option = op
            set = 'set1'
        op_value = random.choice(op_sets[set][option])
        setattr(env.unwrapped._env.task, option, op_value)
        setattr(env2.unwrapped._env.task, option, op_value)
        #env.unwrapped._env.task.box_color=[0,0,1,1]
        #env2.unwrapped._env.task.box_color=[0,1,0,1]
    return env, env2

########################################################################################
# Control modes
########################################################################################

@safe_disconnect
def calibrate(robot: Robot, cfg: CalibrateControlConfig):
    # TODO(aliberts): move this code in robots' classes
    if robot.robot_type.startswith("stretch"):
        if not robot.is_connected:
            robot.connect()
        if not robot.is_homed():
            robot.home()
        return

    arms = robot.available_arms if cfg.arms is None else cfg.arms
    unknown_arms = [arm_id for arm_id in arms if arm_id not in robot.available_arms]
    available_arms_str = " ".join(robot.available_arms)
    unknown_arms_str = " ".join(unknown_arms)

    if arms is None or len(arms) == 0:
        raise ValueError(
            "No arm provided. Use `--arms` as argument with one or more available arms.\n"
            f"For instance, to recalibrate all arms add: `--arms {available_arms_str}`"
        )

    if len(unknown_arms) > 0:
        raise ValueError(
            f"Unknown arms provided ('{unknown_arms_str}'). Available arms are `{available_arms_str}`."
        )

    for arm_id in arms:
        arm_calib_path = robot.calibration_dir / f"{arm_id}.json"
        if arm_calib_path.exists():
            print(f"Removing '{arm_calib_path}'")
            arm_calib_path.unlink()
        else:
            print(f"Calibration file not found '{arm_calib_path}'")

    if robot.is_connected:
        robot.disconnect()

    if robot.robot_type.startswith("lekiwi") and "main_follower" in arms:
        print("Calibrating only the lekiwi follower arm 'main_follower'...")
        robot.calibrate_follower()
        return

    if robot.robot_type.startswith("lekiwi") and "main_leader" in arms:
        print("Calibrating only the lekiwi leader arm 'main_leader'...")
        robot.calibrate_leader()
        return

    # Calling `connect` automatically runs calibration
    # when the calibration file is missing
    robot.connect()
    robot.disconnect()
    print("Calibration is done! You can now teleoperate and record datasets!")


@safe_disconnect
def teleoperate(robot: Robot, cfg: TeleoperateControlConfig):
    control_loop(
        robot,
        control_time_s=cfg.teleop_time_s,
        fps=cfg.fps,
        teleoperate=True,
        display_cameras=cfg.display_cameras,
    )


@safe_disconnect
def record_sim(
    robot: Robot,
    cfg_env: None,
    cfg: RecordControlConfig,
) -> LeRobotDataset:
    # TODO(rcadene): Add option to record logs
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    env = gym.make(
        cfg_env.task, #"gym_aloha/TrossenAIStationaryTransferCube-v0", "gym_aloha/AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=500, #400
        box_size=cfg_env.box_size,
        box_color=cfg_env.box_color,
        box_pos=cfg_env.box_pos,
        tabletop=cfg_env.tabletop,
        backdrop=cfg_env.backdrop,
        lighting=cfg_env.lighting,
        arms_pos=cfg_env.arms_pos,
        arms_ref=cfg_env.arms_ref,
        #render_mode="human"  # This enables the built-in Gymnasium viewer #anr default
    )

    # Load pretrained policy
    if cfg.policy == 'scripted_policy':
        inject_noise = False
        random_paths = False
        policy = PickAndTransferPolicy(inject_noise=inject_noise,box_size=cfg_env.box_size,random_paths=random_paths)
    elif 'trossen' in env.unwrapped.task:
        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
    else:
        policy_cls = get_policy_class(cfg.policy.type)
        policy = policy_cls.from_pretrained(cfg.policy.pretrained_path)
    
    #anr sim uses env, not robot
    #if not robot.is_connected:
    #    robot.connect()

    listener, events = init_keyboard_listener()

    # Execute a few seconds without recording to:
    # 1. teleoperate the robot to move it in starting position if no policy provided,
    # 2. give times to the robot devices to connect and start synchronizing,
    # 3. place the cameras windows on screen
    #enable_teleoperation = policy is None
    #log_say("Warmup record", cfg.play_sounds)
    #warmup_record(robot, events, enable_teleoperation, cfg.warmup_time_s, cfg.display_cameras, cfg.fps)

    #if has_method(robot, "teleop_safety_stop"):
    #    robot.teleop_safety_stop()

    if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee':
        #companion joint environment to the end effector environment
        task="gym_aloha/TrossenAIStationaryTransferCube-v0"
        env2 = gym.make(
            task,
            obs_type="pixels_agent_pos",
            max_episode_steps=500, #400
            box_size=cfg_env.box_size,
            box_color=cfg_env.box_color,
            box_pos=cfg_env.box_pos,
            tabletop=cfg_env.tabletop,
            backdrop=cfg_env.backdrop,
            lighting=cfg_env.lighting,
            arms_pos=cfg_env.arms_pos,
            arms_ref=cfg_env.arms_ref,
             #render_mode="human"  # This enables the built-in Gymnasium viewer #anr default
        )

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        if cfg_env.rand_ops!=None:
            [env, env2]=random_env_options(env,env2,cfg_env.rand_ops)

        print(f'recording episode {recorded_episodes}')
        #log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        episode=[]
        record_sim_episode(
            robot=robot,
            dataset=dataset,
            env=env,
            events=events,
            episode_time_s=cfg.episode_time_s,
            display_cameras=cfg.display_cameras,
            policy=policy,
            fps=cfg.fps,
            single_task=cfg.single_task,
            episode=episode,
        )

        if env.unwrapped.task == 'trossen_ai_stationary_transfer_cube_ee' and len(episode)>0:
            record_sim_episode(
                robot=robot,
                dataset=dataset,
                env=env2,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_cameras=cfg.display_cameras,
                policy=None,
                fps=cfg.fps,
                single_task=cfg.single_task,
                episode=episode,
            )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Current code logic doesn't allow to teleoperate during this time.
        # TODO(rcadene): add an option to enable teleoperation during reset
        # Skip reset for the last episode to be recorded
        #if not events["stop_recording"] and (
        #    (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
        #):
        #    log_say("Reset the environment", cfg.play_sounds)
        #    reset_environment(robot, events, cfg.reset_time_s, cfg.fps)

        #if events["rerecord_episode"]:
        #    log_say("Re-record episode", cfg.play_sounds)
        #    events["rerecord_episode"] = False
        #    events["exit_early"] = False
        #    dataset.clear_episode_buffer()
        #    continue

        if dataset.episode_buffer['size']>0: #env.unwrapped.task == 'trossen_ai_stationary_transfer_cube':
            dataset.save_episode()
        recorded_episodes += 1

        #if events["stop_recording"]:
        #    break

    #log_say("Stop recording", cfg.play_sounds, blocking=True)
    #stop_recording(robot, listener, cfg.display_cameras)

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    #log_say("Exiting", cfg.play_sounds)
    return dataset


@safe_disconnect
def replay(
    robot: Robot,
    cfg_env: None,
    cfg: ReplayControlConfig,
):
    # TODO(rcadene, aliberts): refactor with control_loop, once `dataset` is an instance of LeRobotDataset
    # TODO(rcadene): Add option to record logs

    env = gym.make(
        cfg_env.task,
        obs_type="pixels_agent_pos",
        max_episode_steps=500, #400
        box_size=cfg_env.box_size,
        box_color=cfg_env.box_color,
        box_pos=cfg_env.box_pos,
        tabletop=cfg_env.tabletop,
        backdrop=cfg_env.backdrop,
        lighting=cfg_env.lighting,
        arms_pos=cfg_env.arms_pos,
        arms_ref=cfg_env.arms_ref,
        #render_mode="human"  # This enables the built-in Gymnasium viewer #anr default
    )

    dataset = LeRobotDataset(cfg.repo_id, root=cfg.root, episodes=[cfg.episode])
    actions = dataset.hf_dataset.select_columns("action")

    #anr using env instead of robot
    #if not robot.is_connected:
    #    robot.connect()

    numpy_observation, info = env.reset()

    log_say("Replaying episode", cfg.play_sounds, blocking=True)
    for idx in range(dataset.num_frames):
        start_episode_t = time.perf_counter()

        #visualize:
        display_cameras=True
        if display_cameras and not is_headless():
            cam_list = numpy_observation["pixels"].keys()
            for cam in cam_list:
                image=torch.from_numpy(numpy_observation["pixels"][cam])
                cv2.imshow(cam, cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        action = actions[idx]["action"]
        #robot.send_action(action)

        if isinstance(action, torch.Tensor):
            numpy_action = action.squeeze(0).to("cpu").numpy()
        else:
            numpy_action = action

        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)

        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / cfg.fps - dt_s)

        dt_s = time.perf_counter() - start_episode_t
        log_control_info(robot, dt_s, fps=cfg.fps)


@parser.wrap()
def control_sim_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.env is None:
        raise ValueError(
            "Need a simulated environment use option: --env.type e.g. --env.type=aloha .\n"
        )

    # env = gym.make(
    #     cfg.env.task, #"gym_aloha/TrossenAIStationaryTransferCube-v0", "gym_aloha/AlohaTransferCube-v0",
    #     obs_type="pixels_agent_pos",
    #     max_episode_steps=500, #400
    #     box_size=cfg.env.box_size,
    #     box_color=cfg.env.box_color
    #     #render_mode="human"  # This enables the built-in Gymnasium viewer #anr default
    # )

    robot = make_robot_from_config(cfg.robot)

    if isinstance(cfg.control, CalibrateControlConfig):
        calibrate(robot, cfg.control)
    elif isinstance(cfg.control, TeleoperateControlConfig):
        teleoperate(robot, cfg.control)
    elif isinstance(cfg.control, RecordControlConfig):
        record_sim(robot, cfg.env, cfg.control)
    elif isinstance(cfg.control, ReplayControlConfig):
        replay(robot, cfg.env, cfg.control)
    elif isinstance(cfg.control, RemoteRobotConfig):
        from lerobot.common.robot_devices.robots.lekiwi_remote import run_lekiwi

        run_lekiwi(cfg.robot)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_sim_robot()
