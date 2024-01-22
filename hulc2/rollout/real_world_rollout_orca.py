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
from octo.model.octo_model import OctoModel

N_DIGITS = 7
MAX_REL_POS = 0.02
MAX_REL_ORN = 0.05


def get_ep_start_end_ids(path):
    return np.sort(np.load(Path(path) / "ep_start_end_ids.npy"), axis=0)


def get_frame(path, i):
    filename = Path(path) / f"episode_{i:0{N_DIGITS}d}.npz"
    return np.load(filename, allow_pickle=True)


def reset(env, path, i):
    data = get_frame(path, i)
    robot_state = data["robot_obs"]
    gripper_state = "open" if robot_state[6] > 0.07 else "closed"
    env.reset(
        target_pos=robot_state[:3],
        target_orn=robot_state[3:6],
        gripper_state=gripper_state,
    )


def get_action(path, i, use_rel_actions=False, control_frame="tcp"):
    frame = get_frame(path, i)
    if control_frame == "tcp":
        action = frame["rel_actions_gripper"]
    else:
        print("come here")
        action = frame["rel_actions_world"]
    print(action)
    # pos = action[:3] * MAX_REL_POS
    # orn = action[3:6] * MAX_REL_ORN

    # new_action = {"motion": (pos, orn, action[-1]), "ref": "rel"}
    # return new_action
    return action


def reset(env, path, i):
    data = get_frame(path, i)
    robot_state = data["robot_obs"]
    gripper_state = "open" if robot_state[6] > 0.07 else "closed"
    env.reset(
        target_pos=robot_state[:3],
        target_orn=robot_state[3:6],
        gripper_state=gripper_state,
    )


def load_model(cfg):
    # model = ORCAModel.load_pretrained(cfg.checkpoint_dir)
    # statistics = statistics = ORCAModel.load_dataset_statistics(cfg.checkpoint_dir, "taco_play")
    model = OctoModel.load_pretrained(cfg.checkpoint_dir)
    statistics = model.dataset_statistics

    return model, statistics


def lang_rollout(model, env, statistics, goal, ep_len=500, pred_horizon=1, exp_weight=1, chunk=0):
    print("Type your instruction which the robot will try to follow")
    print(goal)
    task_lang = model.create_tasks(texts=[goal])
    print(task_lang)
    policy_fn = jax.jit(model.sample_actions)
    rollout(env, policy_fn, task_lang, goal, statistics, ep_len=ep_len,
            pred_horizon=pred_horizon, exp_weight=exp_weight, chunk=chunk)


def image_rollout(model, env, statistics, goal_obs=None, ep_len=500, pred_horizon=1, exp_weight=1, chunk=0):
    if goal_obs is None:
        goal_obs = env.get_obs()
    goal_image = goal_obs["rgb_static"]
    goal_image_wrist = goal_obs["rgb_gripper"]
    goal_image_bgr = cv2.cvtColor(goal_image, cv2.COLOR_RGB2BGR)
    goal_image_wrist_bgr = cv2.cvtColor(goal_image_wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("goal_image", goal_image_bgr)
    cv2.imshow("goal_image_gripper", goal_image_wrist_bgr)
    cv2.waitKey()

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
    # task_image = model.create_tasks(goals={"image_0": goal_image})
    policy_fn = jax.jit(model.sample_actions)
    rollout(env, policy_fn, task_image, "image goal", statistics,
            ep_len=ep_len, pred_horizon=pred_horizon, exp_weight=exp_weight, chunk=chunk)


def rollout(env, policy_fn, task, goal, statistics, horizon=2, ep_len=5000, pred_horizon=1, exp_weight=1, chunk=0):
    ewa_action = ExpWeightedAverage(pred_horizon, exp_weight=exp_weight)
    observations = {
        "image_0": np.zeros((1, horizon, 256, 256, 3), dtype=np.uint8),  # batch, horizon, width, height, channels,
        "image_1": np.zeros((1, horizon, 128, 128, 3), dtype=np.uint8),  # for wrist camera
        "timestep_pad_mask": np.array([[True, True]])
    }
    obs_deque = collections.deque(maxlen=horizon)
    obs_deque_wrist = collections.deque(maxlen=horizon)
    # env.reset()
    for i in range(20):
        act = torch.from_numpy(np.array([0, 0, 0., 0, 0, 0, 0]))
        env.step(act)
    # goal_embed = embed_model([goal])[0]
    obs = env.get_obs()
    image = obs['rgb_static']
    image = cv2.resize(image, (256, 256))
    image_wrist = obs['rgb_gripper']
    image_wrist = cv2.resize(image_wrist, (128, 128))
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

    chunk_step_count = 0
    for step in range(ep_len):

        if chunk_step_count == 0:
            actions = policy_fn(observations, task, rng=jax.random.PRNGKey(0))
            pred_actions = (actions[0] * np.array(statistics['action']['std'])) + np.array(statistics['action']['mean'])
            print(pred_actions)
            pred_action_np = np.asarray(pred_actions)
            pred_action_np_chunk = pred_action_np.copy()
            pred_action_np = ewa_action.get_action(pred_action_np)
            pred_action_torch = torch.from_numpy(pred_action_np)
            # transform the gripper action from [0, 1] to [-1, 1]
            # pred_action_torch[0][-1] = pred_action_torch[0][-1] * 2. - 1.
            if chunk_step_count < chunk:
                chunk_step_count += 1
        else:
            pred_action_torch = torch.from_numpy(pred_action_np_chunk[chunk_step_count])
            chunk_step_count += 1
        if chunk_step_count == chunk:
            chunk_step_count = 0

        obs, _, _, _ = env.step(pred_action_torch)

        image = obs['rgb_static']
        image = cv2.resize(image, (256, 256))
        image_wrist = obs['rgb_gripper']
        image_wrist = cv2.resize(image_wrist, (128, 128))
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


class ExpWeightedAverage:
    def __init__(self, pred_horizon: int, exp_weight: int = 0):
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight
        self.act_history = collections.deque(maxlen=self.pred_horizon)

    def get_action(self, actions: np.ndarray) -> np.ndarray:
        assert len(actions) >= self.pred_horizon
        self.act_history.append(actions[:self.pred_horizon])
        num_actions = len(self.act_history)

        curr_act_preds = np.stack([
            pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.act_history)
        ])

        exponents = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = exponents / np.sum(exponents)
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return action


@hydra.main(config_path="../../conf", config_name="inference_real_orca")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    model, statistics = load_model(cfg)
    env = PandaRTXWrapper(env, relative_action=True)

    ep_start_end_ids = get_ep_start_end_ids(cfg.load_dir)
    for start_idx, end_idx in ep_start_end_ids:
        # reset(env, cfg.load_dir, start_idx)
        target_obs = get_frame(cfg.load_dir, end_idx)
        # image_rollout(model, env, statistics, goal_obs=target_obs, ep_len=500, pred_horizon=1, exp_weight=1, chunk=1)

        # goal = "move the slider left"
        goal = "turn on the green light"
        # goal = "turn on the blue light"
        # goal = "turn on the red light"
        # goal = "move the slider right"
        # goal = "stack the blue block on the green block"
        # goal = "unstack the blue block"
        # goal = "open the drawer"

        lang_rollout(model, env, statistics, goal, ep_len=500, pred_horizon=4, exp_weight=1, chunk=1)
        # image_rollout(model, env, statistics, ep_len=500, pred_horizon=1, exp_weight=1, chunk=4)


if __name__ == "__main__":
    main()
