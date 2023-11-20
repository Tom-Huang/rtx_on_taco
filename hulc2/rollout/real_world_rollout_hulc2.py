import os
from pathlib import Path
import time

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from robot_io.utils.utils import FpsController

from hulc2.evaluation.utils import imshow_tensor
from hulc2.models.hulc2 import Hulc2
from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
from hulc2.wrappers.panda_lfp_wrapper import PandaLfpWrapper


train_folder = Path("/export/home/huang/real_world_checkpoints/lang_lfp_single")
checkpoint_name = '17'
checkpoint = get_checkpoints_for_epochs(train_folder, checkpoint_name)[0]
checkpoint = format_sftp_path(checkpoint)
model = Hulc2.load_from_checkpoint(checkpoint)
model.freeze()
model.cuda()

data_dir = Path("/export/home/huang/rtx_test_processed/validation")
ep_range_path = Path("/export/home/huang/rtx_test_processed/validation/ep_start_end_ids.npy")

ep_ranges = np.load(ep_range_path)


with hydra.initialize_config_dir(config_dir="/home/huang/hcg/projects/rtx/code/rtx_on_taco/conf/"):
    cfg = hydra.compose(config_name="inference_real")

robot = hydra.utils.instantiate(cfg.robot)
train_cfg_path = Path(train_folder) / ".hydra/config.yaml"
train_cfg_path = format_sftp_path(train_cfg_path)
train_cfg = OmegaConf.load(train_cfg_path)
train_cfg["datamodule"]["datasets"].pop("lang_dataset", None)
train_cfg.datamodule.root_data_dir = "/export/home/huang/500k_all_tasks_dataset_15hz"
data_module = hydra.utils.instantiate(train_cfg.datamodule, num_workers=0)
data_module.prepare_data()
data_module.setup()
val_dataset_keys = data_module.val_datasets.keys()
print(val_dataset_keys)
dataloader = data_module.val_dataloader()
# dataset = dataloader.dataset.datasets["lang"]
dataset = dataloader.dataset.datasets["vis"]

tmp_env = hydra.utils.instantiate(cfg.env, robot=robot)
env = PandaLfpWrapper(tmp_env, dataset)


def rollout(env, model, goal, ep_len=500):
    #     env.reset()
    obs = env.get_obs()
    model.reset()
    obs = env.get_obs()
    model.replan_freq = 15
#     obs["rgb_obs"] = {"rgb_static": np.expand_dims(obs["rgb_static"], axis=(0,1)), "rgb_gripper": np.expand_dims(obs["rgb_gripper"], axis=(0,1))}
#     obs["depth_obs"] = {"depth_static": np.expand_dims(obs["depth_static"], axis=(0,1)), "depth_gripper": np.expand_dims(obs["depth_gripper"], axis=(0,1))}
#     obs["robot_obs"] = np.expand_dims(obs["robot_state"], axis=0)
    print(obs.keys())
    print(obs["rgb_obs"]["rgb_static"].shape)
    os.makedirs("/tmp/tmp_img", exist_ok=True)
    model.replan_freq = 15
    for step in range(ep_len):
        action = model.step(obs, goal)
        obs, _, _, _ = env.step(action)

        img_tensor = obs["rgb_obs"]["rgb_static"]
        img_tensor = img_tensor.squeeze()

        img = np.transpose(img_tensor.cpu().numpy(), (1, 2, 0))
        cv2.imwrite(f"/tmp/tmp_img/{step:03}.png", img.astype(np.uint8))
        imshow_tensor("rgb_static", obs["rgb_obs"]["rgb_static"] / 255.0 * 2 - 1, wait=1, resize=True)
        k = imshow_tensor("rgb_gripper", obs["rgb_obs"]["rgb_gripper"], wait=1, resize=True)
        # press ESC to stop rollout and return
        if k == 27:
            return


with hydra.initialize_config_dir(config_dir="/home/huang/hcg/projects/rtx/code/rtx_on_taco/conf/model/language_encoder"):
    lang_enc_cfg = hydra.compose(config_name="sbert")
lang_enc = hydra.utils.instantiate(lang_enc_cfg).eval().cuda()

lang_emb = lang_enc(["open the drawer"])
# lang_emb = lang_enc(["lift the blue block"])
# lang_emb = lang_enc(["turn on the green light"])
goal = {"lang": lang_emb}
rollout(env, model, goal, ep_len=500)
