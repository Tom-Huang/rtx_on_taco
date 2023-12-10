# use "orca" environment. Install dependencies of orca, robot_io, and rtx_on_taco in order
from datetime import datetime
import time
import cv2
import hydra
import torch
import tensorflow_hub as hub
from IPython import display
from PIL import Image
import numpy as np
# import rlds
import tensorflow_datasets as tfds
import tensorflow as tf
import sys
from pathlib import Path
import os
import jax
import collections
import click

from robot_io.utils.utils import quat_to_euler
# from calvin_env.utils.utils import angle_between_angles
from hulc2.wrappers.panda_rtx_wrapper import PandaRTXWrapper
from hulc2.evaluation.utils import imshow_tensor
from orca.utils.pretrained_utils import ORCAModel

os.environ['JAX_PLATFORMS'] = 'cpu'  # Force on CPU


def load_model(cfg):
    model = ORCAModel.load_pretrained(cfg.checkpoint_dir)
    statistics = statistics = ORCAModel.load_dataset_statistics(cfg.checkpoint_dir, "taco_play")

    return model, statistics


def lang_rollout(model, env, statistics, goal, ep_len=500):
    print("Type your instruction which the robot will try to follow")
    print(goal)
    task_lang = model.create_tasks(texts=[goal])
    print(task_lang)
    policy_fn = jax.jit(model.sample_actions)
    rollout(env, policy_fn, task_lang, goal, statistics, ep_len=ep_len)


def image_rollout(model, env, statistics, ep_len=500):
    goal_obs = env.get_obs()
    goal_image = goal_obs["rgb_static"]
    goal_image_wrist = goal_obs["rgb_gripper"]

    goal_image = cv2.resize(goal_image, (256, 256))
    goal_image_wrist = cv2.resize(goal_image_wrist, (128, 128))
    goal_image_bgr = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    goal_image_wrist_bgr = cv2.cvtColor(goal_image_wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("goal_image", goal_image_bgr)
    cv2.imshow("goal_image_gripper", goal_image_wrist_bgr)
    cv2.waitKey()
    goal_image = goal_image[None, ...]
    goal_image_wrist = goal_image_wrist[None, ...]
    task_image = model.create_tasks(goals={"image_0": goal_image, "image_1": goal_image_wrist})
    policy_fn = jax.jit(model.sample_actions)
    rollout(env, policy_fn, task_image, "image goal", statistics, ep_len=ep_len)


def rollout(env, policy_fn, task, goal, statistics, horizon=2, ep_len=5000):
    observations = {
        "image_0": np.zeros((1, horizon, 256, 256, 3), dtype=np.uint8),  # batch, horizon, width, height, channels,
        # optionally add "image_1": np.zeros((1, 2, 128, 128,3), dtype=np.uint8) for wrist camera
        "image_1": np.zeros((1, horizon, 128, 128, 3), dtype=np.uint8),  # for wrist camera
        "pad_mask": np.array([[True, True]])
    }
    obs_deque = collections.deque(maxlen=horizon)
    obs_deque_wrist = collections.deque(maxlen=horizon)
    env.reset()
    # goal_embed = embed_model([goal])[0]
    obs = env.get_obs()
    image = obs['rgb_static']
    image = cv2.resize(image, (256, 256))
    image_wrist = obs['rgb_gripper']
    image_wrist = cv2.resize(image, (128, 128))
    # fill the deque with the first observation
    obs_deque.append(image)
    obs_deque.append(image)
    obs_deque_wrist.append(image_wrist)
    obs_deque_wrist.append(image_wrist)

    images = np.stack([x for x in obs_deque])
    observations["image_0"] = images[None, ...]
    assert observations["image_0"].shape[1] == horizon

    images_wrist = np.stack([x for x in obs_deque_wrist])
    observations["image_1"] = images_wrist[None, ...]
    assert observations["image_1"].shape[1] == horizon

    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)

    # dd/mm/YY H:M:S
    goal_str = "_".join(goal.split()) + "_imgs"
    dt_string = now.strftime("%Y_%m_%d_at_%H_%M_%S")
    folder = Path("/tmp") / goal_str / dt_string
    os.makedirs(folder, exist_ok=True)

    for step in range(ep_len):
        actions = policy_fn(observations, task, rng=jax.random.PRNGKey(0))
        pred_actions = (actions[0] * np.array(statistics['action']['std'])) + np.array(statistics['action']['mean'])
        print(pred_actions)
        pred_action_torch = torch.from_numpy(np.asarray(pred_actions))
        print("torch: ", pred_actions)
        # transform the gripper action from [0, 1] to [-1, 1]
        pred_action_torch[0][-1] = pred_action_torch[0][-1] * 2. - 1.
        print("change: ", pred_actions)

        obs, _, _, _ = env.step(pred_action_torch)

        image = obs['rgb_static']
        image = cv2.resize(image, (256, 256))
        image_wrist = obs['rgb_gripper']
        image_wrist = cv2.resize(image, (128, 128))
        # fill the deque with the first observation
        obs_deque.append(image)
        obs_deque_wrist.append(image_wrist)

        images = np.stack([x for x in obs_deque])
        observations["image_0"] = images[None, ...]
        assert observations["image_0"].shape[1] == horizon

        images_wrist = np.stack([x for x in obs_deque_wrist])
        observations["image_1"] = images_wrist[None, ...]
        assert observations["image_1"].shape[1] == horizon

        cv2.imshow("rgb_static", obs["rgb_static"][:, :, ::-1])
        cv2.imshow("rgb_gripper", obs["rgb_gripper"][:, :, ::-1])
        save_path = folder / f"{step:03}.png"
        cv2.imwrite(save_path.as_posix(), obs["rgb_static"][:, :, ::-1])

        k = cv2.waitKey(200)

        # press ESC to stop rollout and return
        if k == 27:
            return


@hydra.main(config_path="../../conf", config_name="inference_real_orca")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    model, statistics = load_model(cfg)
    env = PandaRTXWrapper(env, relative_action=True)

    # goal = "move the slider left"
    goal = "turn on the green light"
    # goal = "turn on the blue light"
    # goal = "turn on the red light"
    # goal = "move the slider right"
    # goal = "stack the blue block on the green block"
    # goal = "unstack the blue block"
    # goal = "open the drawer"

    # lang_rollout(model, env, statistics, goal, ep_len=500)
    image_rollout(model, env, statistics, ep_len=500)


if __name__ == "__main__":
    main()
