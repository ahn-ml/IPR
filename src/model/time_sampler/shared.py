from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
import torch
from torch import device, Tensor

from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class SharedCfg(TwoStageTimeSamplerCfg):
    name: Literal["shared"]


class Shared(TwoStageTimeSampler[SharedCfg]):

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> Float[Tensor, "batch sample height width"]:
        t = self.scalar_time_sampler(
            (batch_size, num_samples, 1, 1), device
        ).expand(-1, -1, *self.resolution)
        return t
    
    def get_normalization_weights(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*#batch"]:
        return torch.ones((1,), device=t.device).expand_as(t)
