from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


TARGETS: tuple[str, ...] = ("Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g")
IDX_COLS: tuple[str, ...] = (
    "image_path",
    "Sampling_Date",
    "State",
    "Species",
    "Pre_GSHH_NDVI",
    "Height_Ave_cm",
)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_SEED: int = 420
DEFAULT_IMG_SIZE: int = 512
DEFAULT_LOSS_WEIGHTS: tuple[float, float, float, float, float] = (0.1, 0.1, 0.1, 0.2, 0.5)


@dataclass(frozen=True)
class ModelConfig:
    head_hidden: int = 1024
    head_depth: int = 2
    head_drop: float = 0.1
    num_neck: int = 0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    batch_size: int = 64
    lr_start: float = 3e-4
    lr_final: float = 5e-5
    wd: float = 1e-2
    early_stopping: int = 15

    swa_epochs: int = 20
    swa_lr: float | None = None
    swa_anneal_epochs: int = 15
    swa_load_best: bool = False
    swa_eval_freq: int = 2


def default_num_workers(reserve: int = 2) -> int:
    import os

    n = (os.cpu_count() or 0) - int(reserve)
    return max(0, n)


def as_tuple_str(xs: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(x) for x in xs)
