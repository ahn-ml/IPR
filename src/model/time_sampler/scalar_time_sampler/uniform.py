from dataclasses import dataclass
from typing import Literal, Sequence

from jaxtyping import Float
import torch
from torch import device, Tensor

from .scalar_time_sampler import ScalarTimeSampler, ScalarTimeSamplerCfg


@dataclass
class UniformCfg(ScalarTimeSamplerCfg):
    name: Literal["uniform"]


class Uniform(ScalarTimeSampler[UniformCfg]):
    
    def __call__(
        self,
        shape: Sequence[int],
        device: device | str = "cpu"
    ) -> Float[Tensor, "*shape"]:
        return torch.rand(shape, device=device)
