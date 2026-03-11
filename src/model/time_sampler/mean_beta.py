from dataclasses import dataclass
from math import prod
from typing import Literal

from jaxtyping import Bool, Float
import torch
from torch import device, Tensor
from torch.distributions.beta import Beta

from .two_stage_time_sampler import TwoStageTimeSampler, TwoStageTimeSamplerCfg


@dataclass
class MeanBetaCfg(TwoStageTimeSamplerCfg):
    name: Literal["mean_beta"]
    beta_sharpness: float = 1.


class MeanBeta(TwoStageTimeSampler[MeanBetaCfg]):
    def __init__(
        self,
        cfg: MeanBetaCfg,
        resolution: tuple[int, int],
    ) -> None:
        super(MeanBeta, self).__init__(cfg, resolution)
        self.dim = prod(resolution)
        self.betas: dict[int, Beta] = {}
        self.init_betas(self.dim)
        
    def init_betas(self, dim: int) -> None:
        if dim > 1 and dim not in self.betas:
            a = b = (dim - 1 - (dim % 2)) ** 1.05 * self.cfg.beta_sharpness
            self.betas[dim] = Beta(a, b)
            half_dim = dim // 2
            self.init_betas(half_dim)
            self.init_betas(dim - half_dim)

    def _get_uniform_l1_conditioned_vector_list(
        self,
        l1_norms: Float[Tensor, "batch"], 
        dim: int,
    ) -> list[Float[Tensor, "batch"]]:
        if dim == 1:
            return [l1_norms]

        device = l1_norms.device
        half_cells = dim // 2

        max_first_contribution = l1_norms.clamp(max=half_cells) # num cells in the first half
        max_second_contribution = l1_norms.clamp(max=dim-half_cells)
        min_first_contribution = (l1_norms - max_second_contribution).clamp_(min=0)

        random_matrix = self.betas[dim].sample((l1_norms.shape[0],)).to(device=device)
        ranges = max_first_contribution - min_first_contribution

        assert ranges.min() >= 0
        first_contribution = min_first_contribution + ranges * random_matrix
        second_contribution = l1_norms - first_contribution

        return self._get_uniform_l1_conditioned_vector_list(first_contribution, half_cells) \
            + self._get_uniform_l1_conditioned_vector_list(second_contribution, dim - half_cells)

    def _sample_time_matrix(
        self, 
        l1_norms: Float[Tensor, "batch"], 
        dim: int
    ) -> Float[Tensor, "batch dim"]:
        vector_list = self._get_uniform_l1_conditioned_vector_list(l1_norms, dim)
        t = torch.stack(vector_list, dim=1)  # [batch_size, dim]
        # shuffle the time matrix (independently for batch elements) to avoid positional biases
        idx = torch.rand_like(t).argsort()
        t = t.gather(1, idx)
        return t

    def get_time_with_mean(
        self,
        mean: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch height width"]:
        shape = mean.shape
        l1_norms = mean.flatten() * self.dim
        t = self._sample_time_matrix(l1_norms, self.dim)
        return t.view(*shape, *self.resolution)

    def get_time(
        self, 
        batch_size: int, 
        num_samples: int = 1,
        device: device | str = "cpu",
    ) -> Float[Tensor, "batch sample height width"]:
        mean = self.scalar_time_sampler((batch_size, num_samples), device)
        return self.get_time_with_mean(mean)
