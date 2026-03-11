from typing import Sequence

from jaxtyping import Float
from torch import Tensor

from .time_scheduler import TimeScheduler
from .graph_sequential import GraphSequential, GraphSequentialCfg


TIME_SCHEDULER = {
    "graph_sequential": GraphSequential,
}


TimeSchedulerCfg = GraphSequentialCfg


def get_time_scheduler(
    cfg: TimeSchedulerCfg,
    max_steps: int,
    patch_size: int,
    patch_grid_shape: Sequence[int],
    dependency_matrix: Float[Tensor, "num_patches num_patches"] | None = None
) -> TimeScheduler:
    return TIME_SCHEDULER[cfg.name](cfg, max_steps, patch_size, patch_grid_shape, dependency_matrix)
