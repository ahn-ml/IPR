from .scalar_time_sampler import ScalarTimeSampler
from .uniform import Uniform, UniformCfg


SCALAR_TIME_SAMPLER = {
    "uniform": Uniform
}


ScalarTimeSamplerCfg = UniformCfg


def get_scalar_time_sampler(
    cfg: ScalarTimeSamplerCfg
) -> ScalarTimeSampler:
    return SCALAR_TIME_SAMPLER[cfg.name](cfg)
