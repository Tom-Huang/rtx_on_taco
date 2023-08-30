import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import kaggle
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, Sampler, SequentialSampler
import torchvision

import hulc2
from hulc2.datasets.utils.episode_utils import load_dataset_statistics
from hulc2.datasets.utils.shared_memory_loader import SharedMemoryLoader
from hulc2.datasets.utils.shared_memory_loader_lang_only import SharedMemoryLoaderLangOnly

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})


class Hulc2RealWorldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(hulc2.__file__).parent / root_data_path
        self.root_data_path = root_data_path

        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.use_shm = np.any(["shm_dataset" in v._target_ for k, v in self.datasets_cfg.items()])
        if "root_data_dirs_list" in kwargs:
            self.root_data_dirs = kwargs["root_data_dirs_list"]

        if "lang_dataset" in self.datasets_cfg:
            split_dict = self._split_lang_dataset(self.root_data_path)
            self.split_dict = split_dict
        elif "vis_dataset" in self.datasets_cfg:
            split_dict = self._split_vis_dataset(self.root_data_path)
            self.split_dict = split_dict
        elif "real_world_imitation_dataset" in self.datasets_cfg:
            split_dict = self._split_imit_dataset(self.root_data_path)
            self.split_dict = split_dict

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.root_data_path.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            logger.info(f"Dataset not found, trying to download from kaggle to {self.root_data_path}")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files("oiermees/taco-robot", path=self.root_data_path, unzip=True)

        if self.use_shm:
            additional_args = {}

        if self.use_shm:
            train_shmem_loader = SharedMemoryLoaderLangOnly(
                datasets_cfg=self.datasets_cfg,
                dataset_dir=self.root_data_path,
                split="training",
                split_dict=self.split_dict,
            )
            val_shmem_loader = SharedMemoryLoaderLangOnly(
                datasets_cfg=self.datasets_cfg,
                dataset_dir=self.root_data_path,
                split="validation",
                split_dict=self.split_dict,
            )

            train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()
            val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

            save_lang_data(train_shm_lookup, val_shm_lookup)

    def setup(self, stage=None):
        # in the real world dataset the splits are not defined over folders, but in a json file
        transforms = load_dataset_statistics(self.root_data_path, self.root_data_path, self.transforms)

        self.train_transforms = {
            cam: [hydra.utils.instantiate(transform, _convert_="partial") for transform in transforms.train[cam]]
            for cam in transforms.train
        }

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform, _convert_="partial") for transform in transforms.val[cam]]
            for cam in transforms.val
        }
        self.train_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()}
        self.val_transforms = {key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()}
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_lang_data()

        for _, dataset in self.datasets_cfg.items():
            if "Dataset" not in dataset._target_:
                continue
            train_dataset = hydra.utils.instantiate(
                dataset,
                datasets_dir=self.root_data_path,
                transforms=self.train_transforms,
                split="training",
                split_dict=self.split_dict,
            )
            val_dataset = hydra.utils.instantiate(
                dataset,
                datasets_dir=self.root_data_path,
                transforms=self.val_transforms,
                split="validation",
                split_dict=self.split_dict,
            )
            if self.use_shm:
                train_dataset.set_lang_data(train_shm_lookup)
                val_dataset.set_lang_data(val_shm_lookup)
            key = dataset.key
            self.train_datasets[key] = train_dataset
            self.val_datasets[key] = val_dataset
            self.modalities.append(key)

    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=False,
                # pin_memory=True,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=False,
                # pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders

    def _split_lang_dataset(self, root_data_path, train_portion=0.9):
        self.lang_folder = self.datasets_cfg.lang_dataset.lang_folder
        lang_data = np.load(root_data_path / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        ids = np.arange(len(lang_data["info"]["indx"]))
        np.random.shuffle(ids)
        val_split_id_start = int(ids.shape[0] * train_portion)

        return {"training": ids[:val_split_id_start], "validation": ids[val_split_id_start:]}

    def _split_vis_dataset(self, root_data_path, train_portion=0.9):
        ep_data = np.load(root_data_path / "ep_start_end_ids.npy")
        ids = np.arange(len(ep_data))
        np.random.shuffle(ids)
        val_split_id_start = int(ids.shape[0] * train_portion)

        return {"training": ids[:val_split_id_start], "validation": ids[val_split_id_start:]}

    def _split_imit_dataset(self, root_data_path, train_portion=0.9):
        self.lang_folder = self.datasets_cfg.real_world_imitation_dataset.lang_folder
        lang_data = np.load(root_data_path / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        ids = np.arange(len(lang_data["info"]["indx"]))
        np.random.shuffle(ids)
        val_split_id_start = int(ids.shape[0] * train_portion)

        return {"training": ids[:val_split_id_start], "validation": ids[val_split_id_start:]}


def save_lang_data(train_shm_lookup, val_shm_lookup):
    save_path = Path("/tmp/") if "TMPDIR" not in os.environ else Path(os.environ["TMPDIR"])
    np.save(save_path / "train_shm_lookup.npy", train_shm_lookup)
    np.save(save_path / "val_shm_lookup.npy", val_shm_lookup)


def load_lang_data():
    load_path = Path("/tmp/") if "TMPDIR" not in os.environ else Path(os.environ["TMPDIR"])
    train_shm_lookup = np.load(load_path / "train_shm_lookup.npy", allow_pickle=True).item()
    val_shm_lookup = np.load(load_path / "val_shm_lookup.npy", allow_pickle=True).item()
    return train_shm_lookup, val_shm_lookup
