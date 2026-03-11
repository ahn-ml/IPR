from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor


FullPrecision = Literal[32, 64, "32-true", "64-true", "32", "64"]
HalfPrecision = Literal[16, "16-true", "16-mixed", "bf16-true", "bf16-mixed", "bf16", "16"]
Stage = Literal["train", "val", "test"]

Parameterization = Literal["eps", "ut"]


# NOTE conditioning both required for model and dataset
# therefore define here to avoid circular dependencies
@dataclass
class ConditioningCfg:
    label: bool = False
    mask: bool = False


class UnbatchedExample(TypedDict, total=False):
    index: int
    image: NotRequired[Float[Tensor, "channel height width"]]
    label: NotRequired[int]
    # if mask == 1: needs to be inpainted
    mask: NotRequired[Float[Tensor, "1 height width"]]
    path: NotRequired[str]


class BatchedExample(TypedDict, total=False):
    index: Int64[Tensor, "batch"]
    image: NotRequired[Float[Tensor, "batch channel height width"]]
    label: NotRequired[Int64[Tensor, "batch"]]
    mask: NotRequired[Float[Tensor, "batch 1 height width"]]
    path: NotRequired[list[str]]


class BatchedEvaluationExample(TypedDict):
    id: str
    data: dict


class SamplingOutput(TypedDict, total=False):
    sample: Float[Tensor, "batch channel height width"]
    all_z_t: NotRequired[list[Float[Tensor, "frame channel height width"]]]
    all_t: NotRequired[list[Float[Tensor, "frame 1 height width"]]]
    all_sigma: NotRequired[list[Float[Tensor, "frame 1 height width"]]]
    all_x: NotRequired[list[Float[Tensor, "frame channel height width"]]]
