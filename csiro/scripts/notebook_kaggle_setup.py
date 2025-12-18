from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from csiro.amp import set_dtype
from csiro.config import DEFAULT_IMG_SIZE, DEFAULT_SEED, IMAGENET_MEAN, IMAGENET_STD, TARGETS
from csiro.data import BiomassBaseCached
from csiro.data import load_train_wide

# This module mirrors the key "globals" cell from `kaggle_CSIRO.ipynb`, but avoids
# doing heavyweight work (like `torch.hub.load`) at import time.

WB = "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"
WL = "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"
WL_plus = "https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"


@dataclass(frozen=True)
class NotebookGlobals:
    model_size: str = "b"
    plus: str = ""
    compile_model: bool = False
    repo_dir: str = "/notebooks/dinov3"
    root: str = "/notebooks/kaggle/csiro"
    img_size: int = DEFAULT_IMG_SIZE
    seed: int = DEFAULT_SEED
    dtype: torch.dtype = torch.bfloat16

    @property
    def csv_path(self) -> str:
        return os.path.join(self.root, "train.csv")

    @property
    def dino_weights(self) -> str:
        return f"/notebooks/dinov3/weights/dinov3_vit{self.model_size}16_pretrain{self.plus}.pth"

    @property
    def hub_model_name(self) -> str:
        return f"dinov3_vit{self.model_size}16{self.plus.replace('_', '')}"


def configure_torch_like_notebook() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def add_dinov3_to_syspath(repo_dir: str) -> None:
    sys.path.insert(0, repo_dir)


def load_model(g: NotebookGlobals, *, verbose: bool = True):
    add_dinov3_to_syspath(g.repo_dir)
    return torch.hub.load(g.repo_dir, g.hub_model_name, source="local", weights=g.dino_weights, verbose=verbose)


def load_wide_df(g: NotebookGlobals):
    return load_train_wide(g.csv_path, root=g.root, targets=TARGETS)


@dataclass(frozen=True)
class NotebookEnv:
    g: NotebookGlobals
    MODEL: Any
    WIDE_DF: Any
    DATASET: Any | None
    NUM_WORKERS: int
    FEAT_DIM: int | None

    REPO_DIR: str
    DINO_WEIGHTS: str
    ROOT: str
    CSV_PATH: str
    TARGETS: Any
    IMAGENET_MEAN: Any
    IMAGENET_STD: Any
    IMG_SIZE: int
    SEED: int
    DTYPE: torch.dtype
    COMPILE_MODEL: bool
    model_size: str
    plus: str

    def as_globals(self) -> dict[str, Any]:
        return {
            "MODEL": self.MODEL,
            "WIDE_DF": self.WIDE_DF,
            "dataset_biomass": self.DATASET,
            "NUM_WORKERS": self.NUM_WORKERS,
            "FEAT_DIM": self.FEAT_DIM,
            "REPO_DIR": self.REPO_DIR,
            "DINO_WEIGHTS": self.DINO_WEIGHTS,
            "ROOT": self.ROOT,
            "CSV_PATH": self.CSV_PATH,
            "TARGETS": self.TARGETS,
            "IMAGENET_MEAN": self.IMAGENET_MEAN,
            "IMAGENET_STD": self.IMAGENET_STD,
            "IMG_SIZE": self.IMG_SIZE,
            "SEED": self.SEED,
            "DTYPE": self.DTYPE,
            "COMPILE_MODEL": self.COMPILE_MODEL,
            "model_size": self.model_size,
            "plus": self.plus,
            "WB": WB,
            "WL": WL,
            "WL_plus": WL_plus,
        }


def setup_env(
    *,
    repo_dir: str = "/notebooks/dinov3",
    root: str = "/notebooks/kaggle/csiro",
    model_size: str = "b",
    plus: str = "",
    img_size: int = DEFAULT_IMG_SIZE,
    seed: int = DEFAULT_SEED,
    dtype: torch.dtype = torch.bfloat16,
    compile_model: bool = False,
    verbose: bool = True,
    make_dataset: bool = False,
    cache_images: bool = True,
) -> NotebookEnv:
    """
    One-call helper for notebooks: sets torch flags + AMP dtype, loads MODEL and WIDE_DF,
    and optionally creates `dataset_biomass`.
    """
    configure_torch_like_notebook()
    set_dtype(dtype)
    g = NotebookGlobals(
        model_size=model_size,
        plus=plus,
        compile_model=compile_model,
        repo_dir=repo_dir,
        root=root,
        img_size=img_size,
        seed=seed,
        dtype=dtype,
    )

    num_workers = max(0, (os.cpu_count() or 0) - 2)
    wide_df = load_wide_df(g)
    model = load_model(g, verbose=verbose)

    feat_dim = getattr(getattr(model, "norm", None), "normalized_shape", [None])[0]
    try:
        feat_dim = int(feat_dim) if feat_dim is not None else None
    except Exception:
        feat_dim = None

    dataset = None
    if make_dataset:
        dataset = BiomassBaseCached(wide_df, img_size=int(img_size), cache_images=bool(cache_images))

    return NotebookEnv(
        g=g,
        MODEL=model,
        WIDE_DF=wide_df,
        DATASET=dataset,
        NUM_WORKERS=num_workers,
        FEAT_DIM=feat_dim,
        REPO_DIR=g.repo_dir,
        DINO_WEIGHTS=g.dino_weights,
        ROOT=g.root,
        CSV_PATH=g.csv_path,
        TARGETS=TARGETS,
        IMAGENET_MEAN=IMAGENET_MEAN,
        IMAGENET_STD=IMAGENET_STD,
        IMG_SIZE=int(img_size),
        SEED=int(seed),
        DTYPE=dtype,
        COMPILE_MODEL=bool(compile_model),
        model_size=str(model_size),
        plus=str(plus),
    )


def setup_into_notebook(
    ns: dict[str, Any] | None = None,
    **kwargs,
) -> NotebookEnv:
    """
    One-call helper that also injects variables into the caller's globals.

    In a notebook:
      from csiro.scripts.notebook_kaggle_setup import setup_into_notebook
      setup_into_notebook()
      MODEL, WIDE_DF, dataset_biomass
    """
    env = setup_env(**kwargs)
    if ns is None:
        import inspect

        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            ns = frame.f_back.f_globals
        else:
            ns = globals()
    ns.update(env.as_globals())
    return env


__all__ = [
    "WB",
    "WL",
    "WL_plus",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TARGETS",
    "NotebookGlobals",
    "configure_torch_like_notebook",
    "add_dinov3_to_syspath",
    "load_model",
    "load_wide_df",
    "NotebookEnv",
    "setup_env",
    "setup_into_notebook",
]
