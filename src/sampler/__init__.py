from typing import Sequence, Union

from jaxtyping import Float
from torch import Tensor

from .sampler import Sampler
from .ipr_sampler import IPRSampler, IPRSamplerCfg

SAMPLER = {
    "ipr_sampler": IPRSampler
}


SamplerCfg = Union[
    IPRSamplerCfg
]


def get_sampler(
    cfg: SamplerCfg,
    patch_size: int | None = None,
    patch_grid_shape: Sequence[int] | None = None,
    dependency_matrix: Float[Tensor, "num_patches num_patches"] | None = None
) -> Sampler:
    return SAMPLER[cfg.name](cfg, patch_size, patch_grid_shape, dependency_matrix)
