import logging
import os
from typing import Any, Dict, Tuple, Union, List
from omegaconf import DictConfig

import gym
import numpy as np
from robot_io.envs.robot_env import RobotEnv
from robot_io.utils.utils import quat_to_euler
import torch

from hulc2.datasets.base_dataset import BaseDataset
from hulc2.datasets.utils.episode_utils import process_depth, process_rgb, process_state

logger = logging.getLogger(__name__)


def obs_dict_to_np(robot_obs):
    tcp_pos = robot_obs["tcp_pos"]
    tcp_orn = quat_to_euler(robot_obs["tcp_orn"])
    gripper_width = robot_obs["gripper_opening_width"]
    gripper_action = 1 if gripper_width > 0.06 else -1
    joint_positions = robot_obs["joint_positions"]

    return np.concatenate([tcp_pos, tcp_orn, [gripper_width], joint_positions, [gripper_action]])


class PandaHulc2Wrapper(gym.Wrapper):
    def __init__(
        self,
        env: RobotEnv,
        observation_space: DictConfig,
        transforms: Dict,
        proprio_state: DictConfig,
        device: str = "cuda:0",
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super(PandaHulc2Wrapper, self).__init__(env)
        self.env = env
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.observation_space_keys = observation_space
        self.transforms = transforms
        self.proprio_state = proprio_state
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"][0]
        logger.info(f"Initialized PandaLfpWrapper for device {self.device}")
        logger.info(f"Relative actions: {self.relative_actions}")

    def transform_vis(self, vis_goal):
        rgb_obs = process_rgb(vis_goal["rgb_obs"], self.observation_space_keys, self.transforms)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        state_obs = process_state(vis_goal, self.observation_space_keys, self.transforms, self.proprio_state)
        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        obs_dict = {**rgb_obs, **state_obs}
        # return rgb_obs
        return obs_dict

    def transform_observation(self, obs):
        # rename keys and concatenate
        _obs = {}
        _obs["robot_obs"] = obs_dict_to_np(obs["robot_state"])
        _obs["rgb_obs"] = {k: obs[k] for k in list(obs.keys()) if k.startswith("rgb")}
        _obs["depth_obs"] = {k: obs[k] for k in list(obs.keys()) if k.startswith("depth")}

        state_obs = process_state(_obs, self.observation_space_keys, self.transforms, self.proprio_state)
        rgb_obs = process_rgb(_obs["rgb_obs"], self.observation_space_keys, self.transforms)
        depth_obs = process_depth(_obs["depth_obs"], self.observation_space_keys, self.transforms)

        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(self.device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})
        obs_dict = {**rgb_obs, **state_obs, **depth_obs}
        obs_dict["robot_obs_raw"] = torch.from_numpy(_obs["robot_obs"]).to(self.device)
        return obs_dict

    def step(self, action_tensor):
        if self.relative_actions:
            action_tensor = torch.clamp(action_tensor, -1, 1)
        action = np.split(action_tensor.squeeze().cpu().detach().numpy(), [3, 6])
        if self.relative_actions:
            # scale actions to metric values
            action[0] *= self.max_rel_pos
            action[1] *= self.max_rel_orn
        action[2] = 1 if action[-1] > 0 else -1
        action_dict = {"motion": action, "ref": "rel" if self.relative_actions else "abs"}
        o, r, d, i = self.env.step(action_dict)

        obs = self.transform_observation(o)
        return obs, r, d, i

    def reset(self, episode=None, robot_obs=None, target_pos=None, target_orn=None, gripper_state="open"):
        if episode is not None:
            robot_obs = episode["state_info"]["robot_obs"][0]

        if robot_obs is not None:
            robot_obs = robot_obs.cpu().numpy()
            target_pos = robot_obs[:3]
            target_orn = robot_obs[3:6]
            gripper_state = "open" if robot_obs[-1] == 1 else "closed"
            obs = self.env.reset(target_pos=target_pos, target_orn=target_orn, gripper_state=gripper_state)
        elif target_pos is not None and target_orn is not None:
            obs = self.env.reset(target_pos=target_pos, target_orn=target_orn, gripper_state=gripper_state)
        else:
            obs = self.env.reset()

        return self.transform_observation(obs)

    def get_obs(self):
        obs = self.env._get_obs()
        return self.transform_observation(obs)
