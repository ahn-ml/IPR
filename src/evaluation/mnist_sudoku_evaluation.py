from dataclasses import dataclass
from typing import Literal

from jaxtyping import Bool, Integer, Shaped
import torch
from torch import Tensor

from ..dataset import DatasetMnistSudoku9x9Eager, DatasetMnistSudoku9x9Lazy

from .mnist_evaluation import MnistEvaluation, MnistEvaluationCfg


@dataclass
class MnistSudokuEvaluationCfg(MnistEvaluationCfg):
    name: Literal["mnist_sudoku"] = "mnist_sudoku"


class MnistSudokuEvaluation(MnistEvaluation[MnistSudokuEvaluationCfg]):
    def __init__(
        self, 
        cfg: MnistEvaluationCfg, 
        tag: str, 
        dataset: DatasetMnistSudoku9x9Eager | DatasetMnistSudoku9x9Lazy,
        patch_size: int | None = None,
        patch_grid_shape: tuple[int, int] | None = None,
        deterministic: bool = False
    ) -> None:
        super().__init__(cfg, tag, dataset, patch_size, patch_grid_shape, deterministic)
    
    def classify(
        self,
        pred: Integer[Tensor, "batch grid_size grid_size"]
    ) -> tuple[
        Bool[Tensor, "batch"],
        dict[str, Shaped[Tensor, "batch"]]
    ]:
        batch_size, grid_size = pred.shape[:2]
        sub_grid_size = round(grid_size ** 0.5)
        dtype, device = pred.dtype, pred.device
        pred = pred - 1 # Shift [1, 9] to [0, 8] for indices
        ones = torch.ones((1,), dtype=dtype, device=device).expand_as(pred)
        dist = torch.zeros((batch_size,), dtype=dtype, device=device)
        for dim in range(1, 3):
            cnt = torch.full_like(pred, fill_value=-1)
            cnt.scatter_add_(dim=dim, index=pred, src=ones)
            dist.add_(cnt.abs_().sum(dim=(1, 2)))
        # Subgrids
        grids = pred.unfold(1, sub_grid_size, sub_grid_size)\
            .unfold(2, sub_grid_size, sub_grid_size).reshape(-1, grid_size, grid_size)
        cnt = torch.full_like(grids, fill_value=-1)
        cnt.scatter_add_(dim=dim, index=grids, src=ones)
        dist.add_(cnt.abs_().sum(dim=(1, 2)))
        label = dist == 0
        return label, {"distance": dist}
