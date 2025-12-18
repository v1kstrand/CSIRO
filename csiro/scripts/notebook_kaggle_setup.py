from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from csiro.amp import set_dtype
from csiro.config import DEFAULT_IMG_SIZE, DEFAULT_SEED, IMAGENET_MEAN, IMAGENET_STD, TARGETS
from csiro.data import BiomassBaseCached, load_train_wide

# Notebook globals copied from `kaggle_CSIRO.ipynb` for convenience.
WB = "https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"
WL = "https://dinov3.llamameta.net/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"
WL_plus = "https://dinov3.llamameta.net/dinov3_vith16plus/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoidW84aXJvdGQyeThwcGpuNXFveGthZTE4IiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NjU5NzI4MTd9fX1dfQ__&Signature=H5H5kLVc6V83i-s2euNHx6t9KlVeG27QKX6qtkXNiLwEzuCshJD4RfwUbQv8oBJOZXPezAVJZPRkYRdsb4jh-LQ72DZtEuNkjNKHf7Pn57wzee0bjEYjWdJmOqK4waaSe9TQqELM%7EPgzdAT4LCSHYcFQ%7EleRnHGGGJiHBmTd6e1xZYhvUCfkvVD1TG-zM7R0-P%7EMLetHMvWl%7EUapCMYthsWqZctsYAQKUQxsLrly8Y4EaM8hm5nowpArPZC4myNO1iiXld5Hc3t9CVLEdYT7LIct0x6cf3-B-6WOgxGb7LdLPCcZPPfoGgX3KGtTAgNQYOpGFs-hgILFHRKVOJ7T3A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1893388161261111"


def setup_into_notebook(
    *,
    ns: dict[str, Any] | None = None,
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
) -> dict[str, Any]:
    if ns is None:
        import inspect

        frame = inspect.currentframe()
        ns = frame.f_back.f_globals if frame and frame.f_back else globals()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_dir))

    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    set_dtype(dtype)

    CSV_PATH = os.path.join(root, "train.csv")
    DINO_WEIGHTS = os.path.join(repo_dir, "weights", f"dinov3_vit{model_size}16_pretrain{plus}.pth")
    hub_name = f"dinov3_vit{model_size}16{plus.replace('_', '')}"

    WIDE_DF = load_train_wide(CSV_PATH, root=root, targets=TARGETS)
    MODEL = torch.hub.load(repo_dir, hub_name, source="local", weights=DINO_WEIGHTS, verbose=verbose)

    FEAT_DIM = getattr(getattr(MODEL, "norm", None), "normalized_shape", [None])[0]
    try:
        FEAT_DIM = int(FEAT_DIM) if FEAT_DIM is not None else None
    except Exception:
        FEAT_DIM = None

    dataset_biomass = None
    if make_dataset:
        dataset_biomass = BiomassBaseCached(WIDE_DF, img_size=int(img_size), cache_images=bool(cache_images))

    env = dict(
        WB=WB,
        WL=WL,
        WL_plus=WL_plus,
        model_size=str(model_size),
        plus=str(plus),
        COMPILE_MODEL=bool(compile_model),
        REPO_DIR=str(repo_dir),
        DINO_WEIGHTS=str(DINO_WEIGHTS),
        MODEL=MODEL,
        NUM_WORKERS=max(0, (os.cpu_count() or 0) - 2),
        ROOT=str(root),
        CSV_PATH=str(CSV_PATH),
        TARGETS=TARGETS,
        WIDE_DF=WIDE_DF,
        IMAGENET_MEAN=IMAGENET_MEAN,
        IMAGENET_STD=IMAGENET_STD,
        IMG_SIZE=int(img_size),
        SEED=int(seed),
        DTYPE=dtype,
        FEAT_DIM=FEAT_DIM,
        dataset_biomass=dataset_biomass,
    )
    ns.update(env)
    return env


setup = setup_into_notebook
setup_env = setup_into_notebook

__all__ = ["setup", "setup_into_notebook", "setup_env", "WB", "WL", "WL_plus"]

