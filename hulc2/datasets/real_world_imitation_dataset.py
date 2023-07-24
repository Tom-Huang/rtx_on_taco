from collections import defaultdict
from itertools import chain
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from omegaconf import ListConfig, OmegaConf
import torch

import hulc2
from hulc2.datasets.base_dataset import BaseDataset, get_validation_window_size
from hulc2.datasets.utils.episode_utils import (
    get_state_info_dict,
    lookup_naming_pattern,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)

# from hulc2.utils.transforms import ActionNormalization

logger = logging.getLogger(__name__)


class RealWorldImitationDataset(BaseDataset):
    def __init__(
        self,
        *args: Any,
        tasks: List[str],
        number_demos: int,
        act_max_bound: Union[List[float], ListConfig],
        act_min_bound: Union[List[float], ListConfig],
        skip_frames: int = 1,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.skip_frames = skip_frames
        (self.episode_lookup, self.episode_len_lookup) = self._build_file_indices_lang_task_demo(  # type: ignore
            self.abs_datasets_dir, tasks, number_demos
        )
        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, "npz")
        self.action_max_bound = act_max_bound
        self.action_min_bound = act_min_bound
        if self.observation_space["actions"][0] == "actions":
            # set action bounds
            self.discrete_gripper = True
            self._setup_action_bounds(self.abs_datasets_dir, act_max_bound, act_min_bound, load_action_bounds=True)
        # self.action_normalization = ActionNormalization(self.action_max_bound, self.action_min_bound)

    def _setup_action_bounds(self, dataset_dir, act_max_bound, act_min_bound, load_action_bounds):
        if load_action_bounds:
            try:
                statistics_path = Path(hulc2.__file__).parent / dataset_dir / "statistics.yaml"
                statistics = OmegaConf.load(statistics_path)
                act_max_bound = statistics.act_max_bound
                act_min_bound = statistics.act_min_bound
                logger.info(f"Loaded action bounds from {statistics_path}")
            except FileNotFoundError:
                logger.info(
                    f"Could not load statistics.yaml in {statistics_path}, taking action bounds defined in hydra conf"
                )
        if self.discrete_gripper:
            self.gripper_bounds = (act_min_bound[-1], act_max_bound[-1])
            act_max_bound = act_max_bound[:-1]  # for discrete grasp
            act_min_bound = act_min_bound[:-1]
        self.action_max_bound = np.array(act_max_bound, dtype=np.float32)
        self.action_min_bound = np.array(act_min_bound, dtype=np.float32)

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def get_window_size(self, idx: int) -> int:
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = (
                self.min_window_size
                + np.nonzero(
                    np.array(self.episode_lookup[idx : idx + window_diff + 1])
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
                - 1
            )
            max_window = min(self.max_window_size, steps_to_next_episode)
        else:
            max_window = self.max_window_size

        if self.validation:
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def zip_sequence(self, start_idx: int, end_idx: int, idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame

        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        # keys.append("scene_obs")
        episodes = [self.load_episode(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        return episode

    def get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.
        Args:
            idx: index of starting frame
            window_size: length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        start_file_indx = self.episode_lookup[idx]
        end_file_indx = start_file_indx + window_size

        episode = self.zip_sequence(start_file_indx, end_file_indx, idx)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore

        return seq_dict

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.
        Args:
            file_idx: index of starting frame.
        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _build_file_indices_lang_task_demo(
        self, abs_datasets_dir: Path, tasks: List[str], number_demos: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.
        It filters the dataset for episodes containing task_id and collects number_demos of them.
        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.
        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_emb = lang_data["language"]["emb"]  # length total number of annotations
        lang_text = lang_data["language"]["ann"]
        lang_task = lang_data["language"]["task"]
        lang_lookup = []
        episode_len_lookup = []
        tasks_counter: Dict = defaultdict(int)
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            task = lang_text[i]
            # Check if the task is part of the desired set of tasks and if enough demonstrations have been collected for this task
            if task not in tasks or tasks_counter[task] >= number_demos:
                continue
            tasks_counter[task] += 1

            # Add the language and episode indices to the lookup lists
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1
            episode_len_lookup.append(end_idx - start_idx)
            if len(episode_len_lookup) >= number_demos * len(tasks):
                break

        return np.array(episode_lookup), np.array(episode_len_lookup)  # , lang_lookup, lang_emb, lang_text
