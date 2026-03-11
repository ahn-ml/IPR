from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Generic, Iterator, Sequence, TypeVar

from jaxtyping import Bool, Float

from torch import device, Tensor


@dataclass
class TimeSchedulerCfg:
    name: str


T = TypeVar("T", bound=TimeSchedulerCfg)


class TimeScheduler(Generic[T], ABC):
    def __init__(
        self,
        cfg: T,
        max_steps: int,
        patch_size: int,
        patch_grid_shape: Sequence[int],
        dependency_matrix: Float[Tensor, "num_patches num_patches"] | None = None
    ) -> None:
        self.cfg = cfg
        self.max_steps = max_steps
        self.patch_size = patch_size
        self.patch_grid_shape = patch_grid_shape
        self.total_patches = prod(patch_grid_shape)
        self.dependency_matrix = dependency_matrix

    @property
    def num_timesteps(self) -> int:
        return self.max_steps

    @abstractmethod
    def __call__(
        self,
        batch_size: int,
        device: device,
        mask: Float[Tensor, "batch 1 height width"] | None = None
    ) -> Iterator[
        tuple[
            Float[Tensor, "batch 1 height width"],
            Float[Tensor, "batch 1 height width"]
        ]
    ]:
        pass
