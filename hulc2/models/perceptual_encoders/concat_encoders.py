from typing import Dict, Optional

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss


class ConcatEncoders(nn.Module):
    def __init__(
        self,
        rgb_static: DictConfig,
        proprio: DictConfig,
        device: torch.device,
        depth_static: Optional[DictConfig] = None,
        rgb_gripper: Optional[DictConfig] = None,
        depth_gripper: Optional[DictConfig] = None,
        tactile: Optional[DictConfig] = None,
        state_decoder: Optional[DictConfig] = None,
        views_num: int = 1,
    ):
        super().__init__()
        self.vn = views_num
        self._latent_size = rgb_static.visual_features * self.vn
        if rgb_gripper:
            self._latent_size += rgb_gripper.visual_features
        if depth_static:
            self._latent_size += depth_static.visual_features * self.vn
        if depth_gripper:
            self._latent_size += depth_gripper.visual_features
        if tactile:
            self._latent_size += tactile.visual_features
        visual_features = self._latent_size
        # super ugly, fix this clip ddp thing in a better way
        if "clip" in rgb_static["_target_"] or "r3m" in rgb_static["_target_"]:
            self.rgb_static_encoder = hydra.utils.instantiate(rgb_static, device=device)
        else:
            self.rgb_static_encoder = hydra.utils.instantiate(rgb_static)
        self.depth_static_encoder = hydra.utils.instantiate(depth_static) if depth_static else None
        self.rgb_gripper_encoder = hydra.utils.instantiate(rgb_gripper) if rgb_gripper else None
        self.depth_gripper_encoder = hydra.utils.instantiate(depth_gripper) if depth_gripper else None
        self.tactile_encoder = hydra.utils.instantiate(tactile)
        self.proprio_encoder = hydra.utils.instantiate(proprio)
        if self.proprio_encoder:
            self._latent_size += self.proprio_encoder.out_features

        self.state_decoder = None
        if state_decoder:
            state_decoder.visual_features = visual_features
            state_decoder.n_state_obs = self.proprio_encoder.out_features
            self.state_decoder = hydra.utils.instantiate(state_decoder)

        self.current_visual_embedding = None
        self.current_state_obs = None

    @property
    def latent_size(self):
        return self._latent_size

    def forward(
        self, imgs: Dict[str, torch.Tensor], depth_imgs: Dict[str, torch.Tensor], state_obs: torch.Tensor
    ) -> torch.Tensor:
        # handle multiview images
        views_num = len([1 for k in imgs.keys() if "rgb_static" in k])
        if views_num > 1:
            # (bs, seq_num, vn, h, w)
            rgb_static = torch.concat([imgs[f"rgb_static_{i}"].unsqueeze(2) for i in range(views_num)], dim=2)
            # (bs, seq_num, vn, h, w)
            depth_static = (
                torch.concat([depth_imgs[f"depth_static_{i}"].unsqueeze(2) for i in range(views_num)], dim=2)
                if "depth_static_0" in depth_imgs
                else None
            )
        else:
            if "rgb_static_0" in imgs:
                rgb_static = imgs["rgb_static_0"]
                depth_static = depth_imgs["depth_static_0"] if "depth_static_0" in depth_imgs else None
            else:
                rgb_static = imgs["rgb_static"]
                depth_static = depth_imgs["depth_static"] if "depth_static" in depth_imgs else None
        rgb_gripper = imgs["rgb_gripper"] if "rgb_gripper" in imgs else None
        rgb_tactile = imgs["rgb_tactile"] if "rgb_tactile" in imgs else None
        depth_gripper = depth_imgs["depth_gripper"] if "depth_gripper" in depth_imgs else None

        if views_num > 1:
            b, s, vn, c, h, w = rgb_static.shape  # (batch_size, seq_num, views_num, channel, height, width)
        else:
            b, s, c, h, w = rgb_static.shape
        rgb_static = rgb_static.reshape(
            -1, c, h, w
        ).contiguous()  # (batch_size * sequence_length, 3, 200, 200) or (batch_size * sequence_length * views_num, 3, 200, 200)
        # ------------ Vision Network ------------ #
        # (batch*seq_len, 64) or (batch_size * sequence_length * views_num, 64)
        encoded_imgs = self.rgb_static_encoder(rgb_static)
        encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64) or (batch, seq, 64 * views_num)

        if depth_static is not None:
            if views_num == 1:
                depth_static = torch.unsqueeze(depth_static, 2)  # (batch_size, sequence_length, 1, 200, 200)
            # (batch_size * sequence_length, 1, 200, 200) or (batch_size * sequence_length * vn, 1, 200, 200)
            depth_static = depth_static.reshape(-1, 1, h, w).contiguous()

            encoded_depth_static = self.depth_static_encoder(depth_static)  # (bs*seq_len, 64) or (bs*seq_len*vn, 64)
            encoded_depth_static = encoded_depth_static.reshape(b, s, -1)  # (bs, seq_len, 64) or (bs, seq_len, 64 * vn)
            encoded_imgs = torch.cat([encoded_imgs, encoded_depth_static], dim=-1)

        if rgb_gripper is not None:
            b, s, c, h, w = rgb_gripper.shape
            rgb_gripper = rgb_gripper.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_imgs_gripper = self.rgb_gripper_encoder(rgb_gripper)  # (batch*seq_len, 64)
            encoded_imgs_gripper = encoded_imgs_gripper.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_gripper], dim=-1)
            if depth_gripper is not None:
                depth_gripper = torch.unsqueeze(depth_gripper, 2)
                depth_gripper = depth_gripper.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                encoded_depth_gripper = self.depth_gripper_encoder(depth_gripper)
                encoded_depth_gripper = encoded_depth_gripper.reshape(b, s, -1)  # (batch, seq, 64)
                encoded_imgs = torch.cat([encoded_imgs, encoded_depth_gripper], dim=-1)

        if rgb_tactile is not None:
            b, s, c, h, w = rgb_tactile.shape
            rgb_tactile = rgb_tactile.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_tactile = self.tactile_encoder(rgb_tactile)
            encoded_tactile = encoded_tactile.reshape(b, s, -1)
            encoded_imgs = torch.cat([encoded_imgs, encoded_tactile], dim=-1)

        self.current_visual_embedding = encoded_imgs
        self.current_state_obs = state_obs  # type: ignore
        if self.proprio_encoder:
            state_obs_out = self.proprio_encoder(state_obs)
            perceptual_emb = torch.cat([encoded_imgs, state_obs_out], dim=-1)
        else:
            perceptual_emb = encoded_imgs

        return perceptual_emb
