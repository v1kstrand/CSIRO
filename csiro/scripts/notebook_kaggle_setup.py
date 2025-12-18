from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from csiro.config import DEFAULT_IMG_SIZE, DEFAULT_SEED, IMAGENET_MEAN, IMAGENET_STD, TARGETS
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
]
