from collections import Counter, defaultdict
from functools import partial
from itertools import chain
import json
import logging
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import re
import signal

import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from tqdm import tqdm

from hulc2.datasets.shm_dataset import ShmDataset
from hulc2.utils.data_utils import get_split_data
from hulc2.utils.split_dataset import get_split_sequences, get_start_end_ids

log = logging.getLogger(__name__)


def gather_results(return_dict):
    episode_lookup_vision = defaultdict(list)
    lang_episode_dict = defaultdict(dict)
    for proc in sorted(return_dict):
        for key in return_dict[proc][0]:
            episode_lookup_vision[key] += return_dict[proc][0][key]
            lang_episode_dict[key].update(return_dict[proc][1][key])
    return episode_lookup_vision, lang_episode_dict


def check_shm_lookup_exists(dataset_type):
    load_path = Path("/tmp/") if "TMPDIR" not in os.environ else Path(os.environ["TMPDIR"])
    try:
        data = np.load(load_path / f"{dataset_type}_shm_lookup.npy", allow_pickle=True).item()
        return data
    except FileNotFoundError:
        return None


class SharedMemoryLoaderCombinedLangOnly:
    def __init__(self, datasets_cfg, dataset_dirs, split, split_dict_list=[], **other_args):
        self.ep_ids_file = "ep_start_end_ids.npy"
        self.load_lang_embeddings = datasets_cfg.lang_dataset.load_lang_embeddings
        self.split = split
        self.split_dict_list = split_dict_list
        self.obs_space = datasets_cfg.lang_dataset.obs_space
        self.dataset_dirs = dataset_dirs
        self.dataset_type = "train" if "training" in split else "val"
        self.lang_folder = datasets_cfg.lang_dataset.lang_folder
        self.save_format = "npz"
        # self.naming_pattern, self.n_digits = self.lookup_naming_pattern()
        pattern_n_digits_pairs = [self.lookup_naming_pattern(i) for i in range(len(self.dataset_dirs))]
        self.naming_patterns = [x[0] for x in pattern_n_digits_pairs]
        self.n_digits = [x[1] for x in pattern_n_digits_pairs]
        if "vision_dataset" in datasets_cfg:
            self.min_window_size_vision = datasets_cfg.vision_dataset.min_window_size
        if "lang_dataset" in datasets_cfg:
            self.min_window_size_lang = datasets_cfg.lang_dataset.min_window_size
            if "vision_dataset" not in datasets_cfg:
                self.min_window_size_vision = self.min_window_size_lang
                self.max_window_size_vision = datasets_cfg.lang_dataset.max_window_size

        self.data_percent = 1.0 if self.dataset_type == "val" else datasets_cfg.lang_dataset.data_percent
        self.n_proc = 8

    def worker_process(self, proc_num, dataset_i, ep_start_end_ids, offsets, shmem, lang_ep_start_end_ids, return_dict):
        episode_lookup_vision = defaultdict(list)
        lang_episode_dict = defaultdict(dict)
        if proc_num == 0:
            pbar = tqdm(total=np.sum(np.diff(ep_start_end_ids)), leave=False)
        else:
            pbar = None
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            seq = self.zip_sequence(dataset_i, start_idx, end_idx, pbar)
            for key, array in seq.items():
                shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shmem[key].buf, offset=offsets[key])
                shared_array[:] = array[:]

                for j, idx in enumerate(range(start_idx, end_idx + 1 - self.max_window_size_vision)):
                    # episode offset values are correct, j is just the relative offset within an episode
                    episode_lookup_vision[key].append(
                        (offsets[key], j)
                    )  # what is j? j range from 0 to end-start+1-min_win_size
                    # when the j is at the beginning of the episode, add it to lang_episode_dict
                    if idx in lang_ep_start_end_ids[:, 0]:
                        lang_episode_dict[key][idx] = (offsets[key], j)
                offsets[key] += array.nbytes
        return_dict[proc_num] = episode_lookup_vision, lang_episode_dict
        if pbar is not None:
            pbar.close()

    def load_data_in_shared_memory(self):
        ep_start_end_ids_list = []
        lang_ep_start_end_ids_list = []
        lang_data_list = []
        lang_ann_list = []
        for dataset_i, dataset_dir in enumerate(self.dataset_dirs):
            lang_data = np.load(dataset_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
            ep_start_end_ids = np.array(lang_data["info"]["indx"])
            ep_start_end_ids_list.append(ep_start_end_ids)

            if dataset_i < len(self.split_dict_list):
                split_dict = self.split_dict_list[dataset_i]
                ep_start_end_ids = ep_start_end_ids[split_dict[self.split]]
                lang_data["info"]["indx"] = ep_start_end_ids
                lang_data["language"]["emb"] = [lang_data["language"]["emb"][x] for x in split_dict[self.split]]
                lang_data["language"]["ann"] = [lang_data["language"]["ann"][i] for i in split_dict[self.split]]
                print(f"##############TOTAL EP NUMBER for dataset {dataset_i}####################")
                print("info_index_num:", ep_start_end_ids.shape)
                print(f"##############TOTAL FRAME NUMBER for dataset {dataset_i}####################")
                print(np.sum([ed - st for (st, ed) in lang_data["info"]["indx"]]))
            else:
                total_len = ep_start_end_ids.shape[0]
                val_split_id_start = int(total_len * 0.9)
                if self.split == "validation":
                    ep_start_end_ids = ep_start_end_ids[val_split_id_start:]
                elif self.split == "training":
                    ep_start_end_ids = ep_start_end_ids[:val_split_id_start]
                lang_data["info"]["indx"] = ep_start_end_ids
            lang_data_list.append(lang_data)

            lang_ep_start_end_ids = np.array(lang_data["info"]["indx"])  # each of them are 64
            lang_ep_start_end_ids_list.append(lang_ep_start_end_ids)
            if self.load_lang_embeddings:
                lang_ann = [x for x in lang_data["language"]["emb"]]
            else:
                lang_ann = lang_data["language"]["ann"]

            lang_ann_list.extend(lang_ann)
        log.info(f"lang_ann_list has total len of {len(lang_ann_list)}")

        shmem, shapes, sizes, dtypes, shmem_lookup = self.create_shmem(ep_start_end_ids_list)

        if shmem_lookup is not None:
            # using existing shared memory
            log.info("Using existing shared memory without reloading it.")
            return shmem_lookup

        lang_lookup = []

        episode_lookup_lang = defaultdict(list)
        dataset_offset_num = 0
        total_size = 0
        episode_lookup_vision = defaultdict(list)
        lang_episode_dict = defaultdict(dict)
        for dataset_i, dataset_dir in enumerate(self.dataset_dirs):
            log.info(
                f"Loading {self.dataset_type} dataset {dataset_i} language episodes into shared memory. "
                f"(progress bar shows only worker process 0)."
            )
            ep_start_end_ids = ep_start_end_ids_list[dataset_i]
            total_size = np.sum(ep_start_end_ids[:, 1] - ep_start_end_ids[:, 0] + len(ep_start_end_ids))

            if self.n_proc > total_size:
                self.n_proc = total_size
            split_indices = np.array_split(ep_start_end_ids, self.n_proc, axis=0)
            split_lens = [np.sum(np.diff(split_indices[i])) for i in range(len(split_indices))]
            obs_size = {key: dtypes[key].itemsize * np.prod(shapes[key]) for key in dtypes}
            offsets = [
                {key: (dataset_offset_num + n) * obs_size[key] for key in dtypes}
                for n in np.cumsum([0] + split_lens[:-1])
            ]
            dataset_offset_num += total_size

            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            processes = []
            # load vision data with multiple processes
            for i in range(self.n_proc):
                p = multiprocessing.Process(
                    target=self.worker_process,
                    args=(
                        i,
                        dataset_i,
                        split_indices[i],
                        offsets[i],
                        shmem,
                        lang_ep_start_end_ids_list[dataset_i],
                        return_dict,
                    ),
                )
                processes.append(p)
                p.start()
            for proc in processes:
                proc.join()

            # collect return_dict for different datasets
            log.info(f"Finish loading dataset {dataset_i} into shared memory.")
            log.info(f"Gathering results from return dict.")
            ith_episode_lookup_vision, ith_lang_episode_dict = gather_results(return_dict)

            log.info(f"Finish gathering results from return dict.")
            for key, value in ith_episode_lookup_vision.items():
                max_lang_dict_id = len(lang_episode_dict[key])
                episode_lookup_vision[key] += value
                for lang_id, lang_offset in ith_lang_episode_dict[key].items():
                    lang_episode_dict[key][max_lang_dict_id + lang_id] = lang_offset

            max_lang_dataset_id = int(np.sum([len(lang_ep_start_end_ids_list[x]) for x in range(dataset_i)]))
            for i, (start_idx, end_idx) in enumerate(tqdm(lang_ep_start_end_ids_list[dataset_i])):
                for key in ith_lang_episode_dict:
                    # offset and step are correct
                    offset, step = ith_lang_episode_dict[key][start_idx]
                    for j, idx in enumerate(range(start_idx, end_idx + 1 - self.min_window_size_lang)):
                        episode_lookup_lang[key].append((offset, step + j))
                for idx in range(start_idx, end_idx + 1 - self.min_window_size_lang):
                    lang_lookup.append(i + max_lang_dataset_id)
        result = {
            "episode_lookup_vision": episode_lookup_vision,
            "episode_lookup_lang": episode_lookup_lang,
            "lang_lookup": lang_lookup,
            "lang_ann": lang_ann_list,
            "shapes": shapes,
            "sizes": sizes,
            "dtypes": dtypes,
        }

        return result

    def create_shmem(self, ep_start_end_ids_list):
        # load first episode to determine memory usage
        seq = self.zip_sequence(0, ep_start_end_ids_list[0][0][0], ep_start_end_ids_list[0][0][0] + 1)
        total_size = np.sum(
            [
                np.sum(ep_start_end_ids[:, 1] - ep_start_end_ids[:, 0] + len(ep_start_end_ids))
                for ep_start_end_ids in ep_start_end_ids_list
            ]
        )
        shmem = {}
        shapes = {}
        sizes = {}
        dtypes = {}

        shm_lookup = check_shm_lookup_exists(self.dataset_type)
        # check if all necessary shared memories are already loaded
        if shm_lookup is not None:
            print("shm_lookup exists")
            try:
                if np.all(
                    [
                        SharedMemory(name=f"{self.dataset_type}_{key}").size == size * total_size
                        for key, size in shm_lookup["sizes"].items()
                    ]
                ):
                    return None, None, None, None, shm_lookup
            except FileNotFoundError as e:
                pass
        for key, array in seq.items():
            try:
                # see if exists
                s = SharedMemory(name=f"{self.dataset_type}_{key}")
                s.close()
                s.unlink()
                log.warning(
                    f"Found existing shared memory {self.dataset_type}_{key}"
                    "In case of multiple training runs on the same node, this will lead to problems."
                )
            except FileNotFoundError:
                pass
            shmem[key] = SharedMemory(create=True, size=array.nbytes * total_size, name=f"{self.dataset_type}_{key}")
            shapes[key] = array.shape[1:]
            sizes[key] = array.nbytes
            dtypes[key] = array.dtype

        # register signal handler for the case that shm data loading process gets interrupted.
        signal.signal(signal.SIGTERM, partial(delete_shm, shmem.keys()))

        return shmem, shapes, sizes, dtypes, None

    def zip_sequence(self, dataset_idx, start_idx, end_idx, pbar=None):
        keys = list(chain(*self.obs_space.values()))
        keys.remove("language")
        # keys.append("scene_obs")
        n_items = end_idx - start_idx
        episode = {}
        data = np.load(self.get_episode_name(dataset_idx, start_idx))
        for key in keys:
            shape = (n_items,) + data[key].shape
            dtype = data[key].dtype
            episode[key] = np.empty(shape=shape, dtype=dtype)
        for i, file_idx in enumerate(range(start_idx, end_idx)):
            with np.load(self.get_episode_name(dataset_idx, file_idx)) as data:
                for key in keys:
                    episode[key][i] = data[key]
            if pbar is not None:
                pbar.update(1)
        return episode

    def get_episode_name(self, dataset_idx, idx):
        """
        Convert frame idx to file name
        """
        return Path(
            f"{self.naming_patterns[dataset_idx][0]}{idx:0{self.n_digits[dataset_idx]}d}{self.naming_patterns[dataset_idx][1]}"
        )

    def lookup_naming_pattern(self, dataset_i, n_digits=None):
        it = os.scandir(self.dataset_dirs[dataset_i])
        while True:
            filename = Path(next(it))
            if self.save_format in filename.suffix and "camera" not in filename.stem:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = n_digits if n_digits is not None else len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits


def delete_shm(shm_keys, signal, frame):
    for dataset_type in ["train", "val"]:
        for shm_key in shm_keys:
            try:
                s = SharedMemory(name=f"{dataset_type}_{shm_key}")
                s.close()
                s.unlink()
                print(f"successfully unlinked {shm_key}")
            except Exception as e:
                print(e)
    exit()


class SignalCallback(Callback):
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if "vis" in trainer.datamodule.train_dataloader() and isinstance(trainer.datamodule.train_dataloader()["vis"].dataset, ShmDataset):  # type: ignore
            shm_keys = trainer.datamodule.train_dataloader()["vis"].dataset.episode_lookup_dict.keys()  # type: ignore
            signal.signal(signal.SIGTERM, partial(delete_shm, shm_keys))
            print("Registered shared memory signal handler.")
