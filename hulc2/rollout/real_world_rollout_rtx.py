from pathlib import Path
import os
# rtx relevant import
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import py_tf_eager_policy


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
import hydra
import cv2
from datetime import datetime

from hulc2.evaluation.utils import imshow_tensor
# from hulc2.models.hulc2 import Hulc2
# from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from hulc2.wrappers.panda_rtx_wrapper import PandaRTXWrapper
from calvin_env.utils.utils import angle_between_angles
from robot_io.utils.utils import quat_to_euler


def load_model(cfg):
    tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        model_path=cfg.checkpoint_dir,
        load_specs_from_pbtxt=True,
        use_tf_function=True)
    print("finish loading policy")

    # test the model with one frame of observation
    observation = {
        'image':
            np.zeros(shape=(256, 320, 3), dtype=np.uint8),
        'natural_language_instruction':
            np.zeros(shape=(), dtype=str),
        'natural_language_embedding':
            np.zeros(shape=(512), dtype=np.float32),
        # 'gripper_closed':
        #     np.zeros(shape=(1), dtype=np.float32),
        # 'height_to_bottom':
        #     np.zeros(shape=(1), dtype=np.float32),
        # 'base_pose_tool_reached':
        #     np.zeros(shape=(7), dtype=np.float32),
        # 'workspace_bounds':
        #     np.zeros(shape=(3, 3), dtype=np.float32),
        # 'orientation_box': np.zeros(shape=(2, 3), dtype=np.float32), 'orientation_start':
        #     np.zeros(shape=(4), dtype=np.float32),
        # 'src_rotation':
        #     np.zeros(shape=(4), dtype=np.float32),
        # 'robot_orientation_positions_box':
        #     np.zeros(shape=(3, 3), dtype=np.float32),
        # 'vector_to_go':
        #     np.zeros(shape=(3), dtype=np.float32),
        # 'rotation_delta_to_go':
        #     np.zeros(shape=(3), dtype=np.float32),
        # 'gripper_closedness_commanded':
        #     np.zeros(shape=(1), dtype=np.float32),
    }

    tfa_time_step = ts.transition(observation, reward=np.zeros(()))

    policy_state = tfa_policy.get_initial_state(batch_size=1)

    action = tfa_policy.action(tfa_time_step, policy_state)
    print(f"RTX model initialized. Action space: {action.action}")
    return tfa_policy


def lang_rollout(model, env, embed_model, goal, ep_len=500):
    print("Type your instruction which the robot will try to follow")
    # while 1:
    #     lang_input = [input("What should I do? \n")]
    #     goal = lang_input[0]
    #     print("sleeping 5 seconds...)")
    #     time.sleep(6)
    #     rollout(env, model, goal, embed_model)
    print(goal)
    rollout(env, model, goal, embed_model, ep_len=ep_len)


def model_step(model, obs, policy_state, goal, goal_embed):
    image = tf.cast(tf.image.resize_with_pad(obs['rgb_static'], target_width=320, target_height=256), np.uint8)
    print(image.shape)
    observation = {
        'image': image.numpy(),
        'natural_language_instruction': goal,
        'natural_language_embedding': goal_embed,
        # 'gripper_closed':
        # np.zeros(shape=(1), dtype=np.float32),
        # 'height_to_bottom':
        #     np.zeros(shape=(1), dtype=np.float32),
        # 'base_pose_tool_reached':
        #     np.zeros(shape=(7), dtype=np.float32),
        # 'workspace_bounds':
        #     np.zeros(shape=(3, 3), dtype=np.float32),
        # 'orientation_box':
        #     np.zeros(shape=(2, 3), dtype=np.float32),
        # 'orientation_start':
        #     np.zeros(shape=(4), dtype=np.float32),
        # 'src_rotation':
        #     np.zeros(shape=(4), dtype=np.float32),
        # 'robot_orientation_positions_box':
        #     np.zeros(shape=(3, 3), dtype=np.float32),
        # 'vector_to_go':
        #     np.zeros(shape=(3), dtype=np.float32),
        # 'rotation_delta_to_go':
        #     np.zeros(shape=(3), dtype=np.float32),
        # 'gripper_closedness_commanded':
        #     np.zeros(shape=(1), dtype=np.float32),
    }
    tfa_time_step = ts.transition(observation, reward=np.zeros(()))
    policy_step = model.action(tfa_time_step, policy_state)

    return policy_step


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


def _unscale_actions_by_bounds(actions, lows, highs, safety_margin=0.01):
    return (actions + 1) * (highs - lows) / 2 + lows


def _unscale_action(action):
    """Rescales actions based on measured per dimension ranges."""
    # Rotation Delta
    rd_lows = tf.constant([-3.2, -0.8, -1.8])
    rd_highs = tf.constant([3.2, 0.2, 2.5])
    action['rotation_delta'] = _unscale_actions_by_bounds(
        action['rotation_delta'], lows=rd_lows, highs=rd_highs
    )

    # World Vector
    wv_lows = tf.constant([0.0, -0.5, 0.0])
    wv_highs = tf.constant([0.8, 0.7, 0.6])
    action['world_vector'] = _unscale_actions_by_bounds(
        action['world_vector'], lows=wv_lows, highs=wv_highs
    )

    return action


def tfa_action_to_bridge_action(tfa_action):
    return np.concatenate((tfa_action['world_vector'], tfa_action['rotation_delta'], tfa_action['gripper_closedness_action']))


def obs_dict_to_np(robot_obs):
    tcp_pos = robot_obs["tcp_pos"]
    tcp_orn = quat_to_euler(robot_obs["tcp_orn"])
    gripper_width = robot_obs["gripper_opening_width"]
    gripper_action = 1 if gripper_width > 0.06 else -1

    return np.concatenate([tcp_pos, tcp_orn, [gripper_action]])


def rollout(env, model, goal, embed_model, ep_len=5000):
    env.reset()
    goal_embed = embed_model([goal])[0]
    obs = env.get_obs()
    policy_state = model.get_initial_state(batch_size=1)
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)

    # dd/mm/YY H:M:S
    goal_str = "_".join(goal.split()) + "_imgs"
    dt_string = now.strftime("%Y_%m_%d_at_%H_%M_%S")
    folder = Path("/tmp") / goal_str / dt_string
    os.makedirs(folder, exist_ok=True)

    for step in range(ep_len):
        policy_step = model_step(model, obs, policy_state, goal, goal_embed)
        action = policy_step.action
        print(action)
        action = _unscale_action(action)
        action = tfa_action_to_bridge_action(action)
        curr_pose = obs_dict_to_np(obs["robot_state"])
        rel_act = to_relative_action(action, curr_pose)
        rel_act_torch = torch.tensor(rel_act)
        # rel_act_torch = torch.tensor(action)

        obs, _, _, _ = env.step(rel_act_torch)
        policy_state = policy_step.state

        cv2.imshow("rgb_static", obs["rgb_static"][:, :, ::-1])
        save_path = folder / f"{step:03}.png"
        cv2.imwrite(save_path.as_posix(), obs["rgb_static"][:, :, ::-1])

        k = cv2.waitKey(200)

        # k = imshow_tensor("rgb_static", obs["rgb_static"], wait=1, resize=True, text=goal)
        # press ESC to stop rollout and return
        if k == 27:
            return


@hydra.main(config_path="../../conf", config_name="inference_real_rtx")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    model = load_model(cfg)
    env = PandaRTXWrapper(env, relative_action=True)
    embed_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
    print("finish loading language encoder")

    goal = "move the slider left"
    # goal = "turn on the green light"
    # goal = "turn on the blue light"
    # goal = "turn on the red light"
    # goal = "move the slider right"
    # goal = "stack the blue block on the green block"
    # goal = "unstack the blue block"
    # goal = "open the drawer"
    lang_rollout(model, env, embed_model, goal, ep_len=500)


if __name__ == "__main__":
    main()
