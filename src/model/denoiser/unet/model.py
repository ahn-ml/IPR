from copy import copy
from dataclasses import dataclass, field, fields
from typing import Literal, Sequence, Type

from jaxtyping import Bool, Float, Int64
from torch import nn, Tensor

from ..activations import Activation
from ..embedding import Embedding
from ..norm_layers import Norm
from .modules import (
    Encoder, 
    Bottleneck, 
    Decoder, 
    ResBlockCfg, 
    MultiHeadAttentionCfg, 
    DownsampleCfg, 
    UpsampleCfg
)
from ..denoiser import Denoiser, DenoiserCfg


@dataclass
class UNetCfg:
    hid_dims: list[int]
    attention: bool | list[bool]
    num_blocks: int | list[int] = 2
    res_block_cfg: ResBlockCfg = field(default_factory=ResBlockCfg)
    attention_cfg: MultiHeadAttentionCfg = field(default_factory=MultiHeadAttentionCfg)
    downsample_cfg: DownsampleCfg = field(default_factory=DownsampleCfg)
    upsample_cfg: UpsampleCfg = field(default_factory=UpsampleCfg)
    out_norm: Norm = "group"
    out_act: Activation = "silu"


class UNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        t_emb: Embedding,
        hid_dims: list[int],
        attention: bool | list[bool],
        num_blocks: int | list[int] = 2,
        res_block_cfg: ResBlockCfg = field(default_factory=ResBlockCfg),
        attention_cfg: MultiHeadAttentionCfg = field(default_factory=MultiHeadAttentionCfg),
        downsample_cfg: DownsampleCfg = field(default_factory=DownsampleCfg),
        upsample_cfg: UpsampleCfg = field(default_factory=UpsampleCfg),
        out_norm: Norm = "group",
        out_act: Activation = "silu",
        bottleneck: bool = True
    ) -> None:
        super(UNet, self).__init__()
        self.encoder = Encoder(
            d_in,
            t_emb,
            hid_dims,
            attention,
            num_blocks,
            res_block_cfg,
            attention_cfg,
            downsample_cfg
        )
        self.bottleneck = Bottleneck(
            self.encoder.out_dim,
            t_emb,
            res_block_cfg,
            attention_cfg
        ) if bottleneck else None

        self.decoder = Decoder(
            self.encoder.channels_list,
            t_emb,
            d_out,
            hid_dims,
            attention,
            num_blocks,
            res_block_cfg,
            attention_cfg,
            upsample_cfg,
            out_norm,
            out_act
        )
        self.init_weights()

    @classmethod
    def from_config(
        cls: Type["UNet"], 
        config: UNetCfg, 
        d_in: int,
        d_out: int,
        t_emb: Embedding,
        bottleneck: bool = True
    ) -> "UNet":
        return cls(
            d_in, d_out, t_emb, 
            bottleneck=bottleneck,
            **{f.name: getattr(config, f.name) for f in fields(config)}
        )

    def init_weights(self) -> None:
        self.encoder.init_weights()
        if self.bottleneck is not None:
            self.bottleneck.init_weights()
        self.decoder.init_weights()

    def forward(
        self, 
        x: Float[Tensor, "batch d_in height width"],
        t: Float[Tensor, "batch 1 height width"],
        c_emb: Float[Tensor, "#batch d_c"] | None = None
    ) -> Float[Tensor, "batch d_out height width"]:
        h, hs = self.encoder(x, t, c_emb)
        if self.bottleneck is not None:
            h = self.bottleneck(h, t, c_emb)
        out = self.decoder(h, hs, t, c_emb)
        return out


@dataclass
class UNetDenoiserCfg(DenoiserCfg, UNetCfg):
    name: Literal["unet"] = "unet"
    t_emb_dim: int = 512


class UNetDenoiser(Denoiser[UNetDenoiserCfg]):
    def __init__(
        self,
        cfg: UNetDenoiserCfg,
        d_in: int,
        d_out: int,
        image_shape: Sequence[int],
        num_classes: int | None = None
    ) -> None:
        super(UNetDenoiser, self).__init__(cfg, d_in, d_out, image_shape, num_classes)
        unet_cfg = copy(cfg)
        unet_cfg.__class__ = UNetCfg
        self.model = UNet.from_config(unet_cfg, d_in, d_out, self.time_embedding)

    @property
    def d_c(self) -> int:
        return self.cfg.t_emb_dim

    def init_weights(self) -> None:
        super(UNetDenoiser, self).init_weights()
        self.model.init_weights()

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
        batch_size, num_times = x.shape[:2]
        
        c_emb = self.embed_conditioning(label)
        if c_emb is not None:
            c_emb = c_emb.unsqueeze(1).expand(-1, num_times, -1).flatten(0, 1)
            
        x = x.flatten(0, 1)
        t = t.flatten(0, 1)
        out = self.model.forward(x, t, c_emb)
        return out.reshape(batch_size, num_times, *out.shape[-3:])
