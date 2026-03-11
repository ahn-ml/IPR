from dataclasses import dataclass
from math import log
from typing import Literal

from jaxtyping import Float
import torch
from torch import Tensor

from .embedding import Embedding, EmbeddingCfg


@dataclass
class EmbeddingSinusodialCfg(EmbeddingCfg):
    name: Literal["sinusodial"]
    # NOTE T_max == 999 in discrete diffusion and here 1
    scale: float | int = 1000
    shift: float | int = -1
    max_period: int = 10000


class EmbeddingSinusodial(Embedding[EmbeddingSinusodialCfg]):

    def __init__(
        self,
        cfg: EmbeddingSinusodialCfg,
        d_out: int
    ):
        super(EmbeddingSinusodial, self).__init__(cfg, d_out)
        self.neg_log_max_period = -log(cfg.max_period)

    def embed(
        self, 
        x: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch d_emb"]:
        """Create embeddings.

        Args:
            x: Data to embed.

        Returns:
            Intermediate embeddings.
        """

        half = self.cfg.d_emb // 2
        freqs = torch.exp(
            self.neg_log_max_period *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half
        ).to(device=x.device)
        args = torch.clamp_min(self.cfg.scale * x.unsqueeze(-1) + self.cfg.shift, 0) * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.cfg.d_emb % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding
