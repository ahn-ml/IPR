from typing import Sequence

from .denoiser import Denoiser
from .unet import UNetDenoiser, UNetDenoiserCfg


DENOISERS = {
    "unet": UNetDenoiser
}

DenoiserCfg = UNetDenoiserCfg


def get_denoiser(
    denoiser_cfg: DenoiserCfg,
    d_in: int,
    d_out: int,
    image_shape: Sequence[int],
    num_classes: int | None = None
) -> Denoiser:
    denoiser: Denoiser = DENOISERS[denoiser_cfg.name](
        denoiser_cfg, d_in, d_out, image_shape, num_classes
    )
    denoiser.init_weights()
    denoiser.freeze()
    return denoiser
