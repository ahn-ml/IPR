from .flow import Flow
from .rectified_flow import RectifiedFlow, RectifiedFlowCfg

from src.type_extensions import Parameterization


FLOW = {
    "rectified": RectifiedFlow
}


FlowCfg = RectifiedFlowCfg


def get_flow(
    cfg: FlowCfg, 
    parameterization: Parameterization
) -> Flow:
    return FLOW[cfg.name](cfg, parameterization)
