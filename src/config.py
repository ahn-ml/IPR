from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .model.wrapper import WrapperCfg
from .type_extensions import FullPrecision, HalfPrecision
from .evaluation import EvaluationCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    every_n_train_steps_persistently: int | None = None
    save_top_k: int | None = None
    resume: bool = False
    save: bool = True


@dataclass
class TrainerCfg:
    max_epochs: int | None = None
    max_steps: int = -1
    val_check_interval: int | float | None = None
    log_every_n_steps: int | None = None
    task_steps: int | None = None
    accumulate_grad_batches: int = 1
    precision: FullPrecision | HalfPrecision | None = None
    num_nodes: int = 1
    validate: bool = True
    profile: bool = False
    detect_anomaly: bool = False


@dataclass
class TorchCfg:
    float32_matmul_precision: Literal["highest", "high", "medium"] | None = None
    cudnn_benchmark: bool = False


@dataclass
class WandbCfg:
    project: str
    entity: str
    activated: bool = True
    mode: Literal["online", "offline", "disabled"] = "online"
    tags: list[str] | None = None


@dataclass
class RootCfg(WrapperCfg):
    mode: Literal["train", "val", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    validation: dict[str, EvaluationCfg]
    test: dict[str, EvaluationCfg]
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    seed: int | None
    mnist_classifier: str | None
    torch: TorchCfg
    wandb: WandbCfg


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )

def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {},
    )
