from abc import ABC
from dataclasses import dataclass, field
from typing import TypeVar

from .scalar_time_sampler import (
    ScalarTimeSampler, 
    ScalarTimeSamplerCfg, 
    get_scalar_time_sampler,
    UniformCfg
)
from .time_sampler import TimeSampler, TimeSamplerCfg


@dataclass
class TwoStageTimeSamplerCfg(TimeSamplerCfg):
    scalar_time_sampler: ScalarTimeSamplerCfg = field(default_factory=lambda: UniformCfg("uniform"))


T = TypeVar("T", bound=TwoStageTimeSamplerCfg)


class TwoStageTimeSampler(TimeSampler[T], ABC):
    scalar_time_sampler: ScalarTimeSampler

    def __init__(
        self,
        cfg: T,
        resolution: tuple[int, int],
    ) -> None:
        super(TwoStageTimeSampler, self).__init__(cfg, resolution)
        self.scalar_time_sampler = get_scalar_time_sampler(cfg.scalar_time_sampler)    
