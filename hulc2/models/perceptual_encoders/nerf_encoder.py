import logging
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.nn import functional as F
import functools
import numpy as np


class PixelNeRFEncoder(torch.nn.Module):
    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        visual_features=64,
        load_ckpt_path="",
        freeze_backbone=True,
    ):
        super().__init__()
        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = self.get_norm_layer(norm_type)

        print("Using torchvision", backbone, "encoder")
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained, norm_layer=norm_layer)
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]
        for param in self.model.parameters():
            param.requires_grad = False
        if not freeze_backbone:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        # if not freeze_backbone:
        #     for param in self.model.parameters():
        #         param.requires_grad = True

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer("latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, visual_features)
        # self.latent (B, L, H, W)
        if load_ckpt_path:
            pretrained_model_ckpts = torch.load(load_ckpt_path)  # , map_location="cuda:0"
            print(pretrained_model_ckpts.keys())
            encoder_state_dict = self.state_dict()
            print(encoder_state_dict.keys())
            print(encoder_state_dict["model.conv1.weight"][0])
            partial_pretrained_dict = {
                ".".join(k.split(".")[1:]): v
                for k, v in pretrained_model_ckpts.items()
                if ".".join(k.split(".")[1:]) in encoder_state_dict
            }
            print(partial_pretrained_dict["model.conv1.weight"][0])
            encoder_state_dict.update(partial_pretrained_dict)
            print(encoder_state_dict["model.conv1.weight"][0])
            self.load_state_dict(encoder_state_dict)
            encoder_state_dict = self.state_dict()
            print(encoder_state_dict["model.conv1.weight"][0])
            logging.info("Pretrained NeRF encoder loaded")

    def get_norm_layer(self, norm_type="instance", group_norm_groups=32):
        """Return a normalization layer
        Parameters:
            norm_type (str) -- the name of the normalization layer: batch | instance | none
        For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
        For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
        """
        if norm_type == "batch":
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_type == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == "group":
            norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
        elif norm_type == "none":
            norm_layer = None
        else:
            raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
        return norm_layer

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size) # (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        self.latent_global = self.latent.view(self.latent.shape[0], self.latent.shape[1], -1).mean(
            dim=-1
        )  # (B, latent_size)
        output = F.relu(self.fc1(self.latent_global))  # batch, 256
        output = self.fc2(output)  # batch, 64
        return output


def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


MODEL_PATH = "/home/huang/hcg/projects/nerf/data/pixelnerf_ckpts/checkpoints/open_the_drawer_nv2/pixel_nerf_latest"

TEST_FRAME_PATH = "/export/home/huang/multiview_processed_15hz/open_the_drawer_processed_15hz/episode_000000.npz"


def main():
    pretrained_model_ckpts = torch.load(MODEL_PATH)
    print(pretrained_model_ckpts.keys())
    encoder = PixelNeRFEncoder()
    encoder_state_dict = encoder.state_dict()
    print(encoder_state_dict.keys())
    print(encoder_state_dict["model.conv1.weight"][0])
    partial_pretrained_dict = {
        ".".join(k.split(".")[1:]): v
        for k, v in pretrained_model_ckpts.items()
        if ".".join(k.split(".")[1:]) in encoder_state_dict
    }
    print(partial_pretrained_dict["model.conv1.weight"][0])
    encoder_state_dict.update(partial_pretrained_dict)
    print(encoder_state_dict["model.conv1.weight"][0])
    encoder.load_state_dict(encoder_state_dict)
    data = dict(np.load(TEST_FRAME_PATH, allow_pickle=True))

    for k, v in data.items():
        if "rgb_static" not in k:
            continue

        # img_t = torch.from_numpy(v)
        preprocess = get_image_to_tensor_balanced()
        img_t = preprocess(v).unsqueeze(0)

        pix_feats = encoder(img_t)
        print(pix_feats.shape)

    pass


if __name__ == "__main__":
    main()
