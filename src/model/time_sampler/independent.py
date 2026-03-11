from dataclasses import dataclass
from typing import Literal

from jaxtyping import Bool, Float
import torch
from torch import device, Tensor

from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class IndependentCfg(TwoStageTimeSamplerCfg):
    name: Literal["independent"]


class Independent(TwoStageTimeSampler[IndependentCfg]):

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> Float[Tensor, "batch sample height width"]:
        return self.scalar_time_sampler(
            (batch_size, num_samples, *self.resolution), 
            device
        )
    
    def get_normalization_weights(
        self, 
        t: Float[Tensor, "*batch"],
        mask: Bool[Tensor, "*#batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        return torch.ones((1,), device=t.device).expand_as(t)
