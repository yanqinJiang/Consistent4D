from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.utils.typing import *


@threestudio.register("latent-solid-color-background")
class LatentSolidColorBackground(BaseBackground):
    @dataclass
    class Config(BaseBackground.Config):
        n_output_dims: int = 4
        latent_white_path: str = "none"
        learned: bool = False

    cfg: Config

    def configure(self) -> None:
        self.env_color: Float[Tensor, "Nc"]
        latent_white = torch.load(self.cfg.latent_white_path)
        latent_white = latent_white.mean(0)
        if self.cfg.learned:
            self.env_color = nn.Parameter(
                latent_white
            )
        else:
            self.register_buffer(
                "env_color", latent_white
            )

    def forward(self, dirs: Float[Tensor, "*B 3"]) -> Float[Tensor, "*B Nc"]:
        
        repeat_time = dirs.shape[0] // (32*32) # latent shape fixed
        latent_bg = self.env_color.permute(1, 2, 0).reshape(-1, 4).repeat(repeat_time, 1)

        return latent_bg 
