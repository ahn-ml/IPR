from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Sequence, TypeVar

from jaxtyping import Bool, Float, Int64
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from src.env import DEBUG
from src.misc.nn_module_tools import freeze
from .class_embedding import get_class_embedding, ClassEmbeddingCfg, ClassEmbeddingParametersCfg
from .embedding import get_embedding, EmbeddingCfg, EmbeddingSinusodialCfg


@dataclass
class DenoiserFreezeCfg:
    time_embedding: bool = False
    class_embedding: bool = False


F = TypeVar("F", bound=DenoiserFreezeCfg)


@dataclass
class DenoiserCfg:
    time_embedding: EmbeddingCfg = field(default_factory=EmbeddingSinusodialCfg)
    class_embedding: ClassEmbeddingCfg = field(default_factory=ClassEmbeddingParametersCfg)
    freeze: F = field(default_factory=DenoiserFreezeCfg)


T = TypeVar("T", bound=DenoiserCfg)


class Denoiser(Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self,
        cfg: T,
        d_in: int,
        d_out: int,
        image_shape: Sequence[int],
        num_classes: int | None = None
    ) -> None:
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.d_out = d_out
        self.image_shape = tuple(image_shape)
        self.labels = num_classes is not None
        self.time_embedding = get_embedding(cfg.time_embedding, self.d_t)
        if self.labels:
            assert cfg.class_embedding is not None
            self.class_embedding = get_class_embedding(cfg.class_embedding, self.d_c, num_classes)

    @property
    @abstractmethod
    def d_c(self) -> int:
        pass
    
    @property
    def d_t(self) -> int:
        return self.d_c

    def freeze(self) -> None:
        if self.cfg.freeze.time_embedding:
            freeze(self.time_embedding, eval=False)
        if self.cfg.freeze.class_embedding:
            freeze(self.class_embedding, eval=False)

    def on_sampling_start(self) -> None:
        # Hook for start of sampling
        return

    def on_sampling_end(self) -> None:
        # Hook for end of sampling
        return

    def embed_conditioning(
        self,
        label: Int64[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "#batch d_c"] | None:
        emb = None
        if self.labels:
            assert label is not None
            emb = self.class_embedding.forward(label)
        return emb

    @abstractmethod
    def forward(
        self, 
        x: Float[Tensor, "batch time d_in height width"],
        t: Float[Tensor, "batch time 1 height width"],
        label: Int64[Tensor, "batch"] | None = None
    ) -> Float[Tensor, "batch time d_out height width"]:
        """
        Arguments:
            x: Input samples
            t: Timesteps
        """
        pass

    @torch.compile(disable=DEBUG)
    def forward_compiled(
        self, 
        x: Float[Tensor, "batch time d_in height width"],
        t: Float[Tensor, "batch time 1 height width"],
        label: Int64[Tensor, "batch"] | None = None,
    ) -> Float[Tensor, "batch time d_out height width"]:
        return self.forward(x, t, label)

    def init_weights(self) -> None:
        if self.labels:
            self.class_embedding.init_weights()

    def get_weight_decay_parameter_groups(self) -> tuple[list[Parameter], list[Parameter]]:
        return list(self.parameters()), []
