from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from math import prod, exp
from typing import Generic, Sequence, TypeVar

from einops import rearrange
from jaxtyping import Float
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn.functional import conv2d
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms.v2 import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RGB,
    ToTensor,
)
from torchvision.transforms import ToTensor

from .types import Example
from ..type_extensions import (
    ConditioningCfg,
    Stage,
    UnbatchedExample
)


@dataclass
class DatasetCfg:
    image_shape: Sequence[int]
    subset_size: int | None
    augment: bool
    grayscale: bool
    root: Path | str = ""
    dependency_matrix_sigma: float = 2.0 # Might make sense to increase for larger resolutions


T = TypeVar("T", bound=DatasetCfg)


class Dataset(TorchDataset, Generic[T], ABC):
    includes_download: bool = False
    num_classes: int | None = None
    cfg: T
    conditioning_cfg: ConditioningCfg
    stage: Stage

    def __init__(
        self,
        cfg: T,
        conditioning_cfg: ConditioningCfg,
        stage: Stage,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.conditioning_cfg = conditioning_cfg
        self.stage = stage
        # Define transforms
        transforms = [
            Lambda(lambda pil_image: self.relative_resize(pil_image, self.cfg.image_shape)),
            CenterCrop(self.cfg.image_shape),
            ToTensor()
        ]
        if self.cfg.augment:
            transforms.insert(2, RandomHorizontalFlip())
        self.transform = Compose(transforms)
        self.rgb_transform = Compose([
            Grayscale() if cfg.grayscale else RGB(),
            Normalize(mean=self.d_data * [0.5], std=self.d_data * [0.5], inplace=True)
        ])
        
    def _get_dependency_2d_gaussian_kernel(
        self,
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "max_grid_size max_grid_size"]:
        sigma = self.cfg.dependency_matrix_sigma
        
        kernel_size = max(grid_shape)
        
        # make sure kernel size is odd
        kernel_size += 1 - kernel_size % 2
        
        kernel_1d = torch.tensor([exp(-(x - kernel_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(kernel_size)])
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()  # Normalize
        
        assert kernel_2d.shape == (kernel_size, kernel_size)
        return kernel_2d 

    def get_dependency_matrix(
        self,
        grid_shape: tuple[int, int]
    ) -> Float[Tensor, "num_patches num_patches"] | None:
        "The Default dependency matrix is based on the locality assumption"
        kernel = self._get_dependency_2d_gaussian_kernel(grid_shape)
        
        total_patches = prod(grid_shape)
        dep_matrix = torch.eye(total_patches) # Initialy all patches are independent
        
        # (x1 y1) 1 x2 y2 as we want to blur over the last two dimensions, 
        # we have to add a dummy dimension for channels
        dep_tensor = rearrange(
            dep_matrix, 
            "... (x y) -> ... 1 x y", 
            x=grid_shape[0], 
            y=grid_shape[1],
        ) 
        blurred = conv2d(dep_tensor, kernel[None, None], padding="same")
        return rearrange(blurred, "... 1 x y -> ... (x y)")

    @property
    def d_data(self) -> int:
        return 1 if self.cfg.grayscale else 3

    @staticmethod
    def relative_resize(
        image: Image.Image, 
        target_shape: Sequence[int]
    ) -> Image.Image:
        target_shape = np.asarray(target_shape[::-1])
        while np.all(np.asarray(image.size) >= 2 * target_shape):
            image = image.resize(
                tuple(x // 2 for x in image.size), 
                resample=Image.Resampling.BOX
            )

        scale = np.max(target_shape / np.asarray(image.size))
        image = image.resize(
            tuple(round(x * scale) for x in image.size), 
            resample=Image.Resampling.BICUBIC
        )
        return image

    @staticmethod
    def concat_mask(
        image: Image.Image,
        mask: Image.Image | Float[np.ndarray, "height width"]
    ) -> Image.Image:
        assert image.mode in ("L", "RGB")
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(np.uint8(255 * mask), mode="L")
        else:
            assert mask.mode == "L"
        if image.mode == "L":
            return Image.merge("LA", (image, mask))
        r, g, b = image.split()
        return Image.merge("RGBA", (r, g, b, mask))

    @abstractmethod
    def load(self, idx: int, **kwargs) -> Example:
        """
        NOTE image of Example must include a mask as alpha channel 
        (LA or RGBA mode) if conditioning_cfg.mask
        """
        pass

    def __getitem__(self, idx: int, **load_kwargs) -> UnbatchedExample:
        sample = self.load(idx, **load_kwargs)
        res = UnbatchedExample(index=idx)
        is_mask_given = sample["image"].mode in ("LA", "RGBA")
        assert not self.conditioning_cfg.mask or is_mask_given, "Mask conditioning but no mask given"
        res["image"] = self.transform(sample["image"])
        if is_mask_given:
            res["mask"] = res["image"][-1:]
            res["image"] = res["image"][:-1]
            res["mask"].round_()
        res["image"] = self.rgb_transform(res["image"])
        if "path" in sample:
                res["path"] = sample["path"]
        if self.conditioning_cfg.label:
            res["label"] = sample["label"]
        return res

    @property
    @abstractmethod
    def _num_available(self) -> int:
        pass

    def __len__(self) -> int:
        if self.cfg.subset_size is not None:
            return self.cfg.subset_size
        return self._num_available
