from .time_sampler import TimeSampler
from .independent import Independent, IndependentCfg
from .mean_beta import MeanBeta, MeanBetaCfg
from .shared import Shared, SharedCfg


TIME_SAMPLER = {
    "independent": Independent,
    "mean_beta": MeanBeta,
    "shared": Shared,
}


TimeSamplerCfg = IndependentCfg | MeanBetaCfg | SharedCfg


def get_time_sampler(
    cfg: TimeSamplerCfg, resolution: tuple[int, int]
) -> TimeSampler:
    return TIME_SAMPLER[cfg.name](cfg, resolution)
