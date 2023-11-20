import os
from pathlib import Path
import time

import torchvision
import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_io.utils.utils import FpsController

from hulc2.evaluation.utils import imshow_tensor
from hulc2.models.hulc2 import Hulc2
from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from hulc2.datasets.utils.episode_utils import load_dataset_statistics
from hulc2.wrappers.panda_hulc2_wrapper import PandaHulc2Wrapper
from hulc2.affordance.models.language_encoders.sbert_lang_encoder import SBertLang


def load_train_cfg(cfg):
    train_cfg_path = Path(cfg.train_folder) / ".hydra/config.yaml"
    train_cfg_path = format_sftp_path(train_cfg_path)
    train_cfg = OmegaConf.load(train_cfg_path)
    train_cfg.datamodule.root_data_dir = cfg.datamodule.root_data_dir

    return train_cfg


def load_model(cfg):
    checkpoint = get_checkpoints_for_epochs(Path(cfg.train_folder), [cfg.checkpoint])[0]
    checkpoint = format_sftp_path(checkpoint)
    model = Hulc2.load_from_checkpoint(checkpoint)
    model.freeze()
    model.cuda()

    train_cfg = load_train_cfg(cfg)

    # with hydra.initialize_config_dir(config_dir="/home/huang/hcg/projects/rtx/code/rtx_on_taco/conf/"):
    #     cfg = hydra.compose(config_name="inference_real")

    # train_cfg["datamodule"]["datasets"].pop("lang_dataset", None)
    # train_cfg.datamodule.root_data_dir = "/export/home/huang/500k_all_tasks_dataset_15hz/training"
    data_module = hydra.utils.instantiate(train_cfg.datamodule, num_workers=0)
    # data_module.prepare_data()
    data_module.use_shm = False
    data_module.setup()

    # in the real world dataset the splits are not defined over folders, but in a json file
    transforms = load_dataset_statistics(
        data_module.root_data_paths[0], data_module.root_data_paths[0], data_module.transforms)

    train_transforms = {
        cam: [hydra.utils.instantiate(transform, _convert_="partial") for transform in transforms.train[cam]]
        for cam in transforms.train
    }

    val_transforms = {
        cam: [hydra.utils.instantiate(transform, _convert_="partial") for transform in transforms.val[cam]]
        for cam in transforms.val
    }
    train_transforms = {key: torchvision.transforms.Compose(val) for key, val in train_transforms.items()}
    val_transforms = {key: torchvision.transforms.Compose(val) for key, val in val_transforms.items()}
    return model, val_transforms


def rollout(env, model, goal, ep_len=500):
    env.reset()
    obs = env.get_obs()
    model.reset()
    obs = env.get_obs()
    model.replan_freq = 15
    print(obs.keys())
    print(obs["rgb_obs"]["rgb_static"].shape)
    os.makedirs("/tmp/tmp_img", exist_ok=True)
    model.replan_freq = 15
    for step in range(ep_len):
        action = model.step(obs, goal)
        print(action)
        obs, _, _, _ = env.step(action)

        img_tensor = obs["rgb_obs"]["rgb_static"]
        img_tensor = img_tensor.squeeze()

        img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
        cv2.imwrite(f"/tmp/tmp_img/{step:03}.png", img.astype(np.uint8))
        k = imshow_tensor("rgb_static", obs["rgb_obs"]["rgb_static"] / 255., wait=1, resize=True)
        # k = imshow_tensor("rgb_gripper", obs["rgb_obs"]["rgb_gripper"], wait=1, resize=True)
        # press ESC to stop rollout and return
        if k == 27:
            return


# train_folder = Path("/export/home/huang/bc_static_taco_lang_checkpoints")
# checkpoint_name = '358'
# train_folder = Path("/export/home/huang/bc_static_taco_extra_lang_checkpoints/r3m_backbone")
# checkpoint_name = '286'


@hydra.main(config_path="../../conf", config_name="inference_real_bc")
def main(cfg):
    model, val_transforms = load_model(cfg)
    train_cfg = load_train_cfg(cfg)

    # setup the environment
    robot = hydra.utils.instantiate(cfg.robot)
    tmp_env = hydra.utils.instantiate(cfg.env, robot=robot)
    env = PandaHulc2Wrapper(tmp_env, train_cfg.datamodule.observation_space,
                            val_transforms, train_cfg.datamodule.proprioception_dims)

    lang_enc = SBertLang("paraphrase-MiniLM-L3-v2", freeze_backbone=True).eval().cuda()

    lang_emb = lang_enc(["turn on the green light"])
    # lang_emb = lang_enc(["turn on the blue light"])
    # lang_emb = lang_enc(["turn on the red light"])
    # lang_emb = lang_enc(["move the slider right"])
    # lang_emb = lang_enc(["stack the blue block on the green block"])
    # lang_emb = lang_enc(["unstack the blue block"])
    # lang_emb = lang_enc(["open the drawer"])

    goal = {"lang": lang_emb}
    rollout(env, model, goal, ep_len=500)


if __name__ == "__main__":
    main()
