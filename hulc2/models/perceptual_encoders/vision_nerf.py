import logging
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.nn import functional as F
import functools
import numpy as np


class VisionNeRF(nn.Module):
    def __init__(
        self,
        visual_features: int,
        load_ckpt_path="",
        freeze_backbone: bool = True,
    ):
        super(VisionNeRF, self).__init__()
        # Load pre-trained R3M resnet-18
        self.model = torchvision.models.resnet18(pretrained=True)
        modules = list(self.model.children())[:-1]
        if load_ckpt_path:
            pretrained_model_ckpts = torch.load(load_ckpt_path)  # , map_location="cuda:0"
            # print(pretrained_model_ckpts.keys())
            encoder_state_dict = self.state_dict()
            # print(encoder_state_dict.keys())
            # print(encoder_state_dict["model.conv1.weight"][0])
            partial_pretrained_dict = {
                ".".join(k.split(".")[1:]): v
                for k, v in pretrained_model_ckpts.items()
                if ".".join(k.split(".")[1:]) in encoder_state_dict
            }
            # print(partial_pretrained_dict.keys())
            # print(partial_pretrained_dict["model.conv1.weight"][0])
            encoder_state_dict.update(partial_pretrained_dict)
            # print(encoder_state_dict["model.conv1.weight"][0])
            self.load_state_dict(encoder_state_dict)
            encoder_state_dict = self.state_dict()
            # print(encoder_state_dict["model.conv1.weight"][0])
            logging.info("Pretrained NeRF encoder loaded")
        # set all grads to false
        for param in self.model.parameters():
            param.requires_grad = False
        if not freeze_backbone:
            # finetune last layer
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        self.model = nn.Sequential(*modules)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model(x)  # batch, 512, 1, 1
        # Add fc layer for final prediction
        x = torch.flatten(x, start_dim=1)  # batch, 512
        output = F.relu(self.fc1(x))  # batch, 256
        output = self.fc2(output)  # batch, 64
        return output
