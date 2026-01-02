from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


from .config import IDX_COLS, TARGETS
from .transforms import TTABatch, post_tfms


def _to_abs_path(root: str | None, p: str) -> str:
    if os.path.isabs(p) or root is None:
        return p
    return os.path.join(root, p)


def load_train_wide(
    csv_path: str,
    *,
    root: str | None = None,
    targets: Sequence[str] = TARGETS,
    idx_cols: Sequence[str] = IDX_COLS,
    image_path_col: str = "image_path",
    target_name_col: str = "target_name",
    target_col: str = "target",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    wide = (
        df.pivot_table(index=list(idx_cols), columns=target_name_col, values=target_col, aggfunc="first")
        .reset_index()
    )
    for t in targets:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide.dropna(subset=list(targets)).reset_index(drop=True)
    wide["abs_path"] = wide[image_path_col].apply(lambda p: _to_abs_path(root, p))
    return wide


class BiomassBaseCached(Dataset):
    def __init__(
        self,
        wide_df: pd.DataFrame,
        *,
        targets: Sequence[str] = TARGETS,
        img_size: int = 512,
        cache_images: bool = True,
        pad_to_square: bool = True,
        pad_fill: int = 0,
    ):
        self.df = wide_df.reset_index(drop=True)
        y = self.df[list(targets)].values.astype(np.float32)
        self.y_log = np.log1p(y)
        self.targets = list(targets)

        if pad_to_square:
            from .transforms import PadToSquare

        self._pre = T.Compose(
            [
                T.Lambda(lambda im: im.convert("RGB")),
                PadToSquare(fill=pad_fill) if pad_to_square else T.Lambda(lambda x: x),
                T.Resize((img_size, img_size), antialias=True),
            ]
        )
        self.cache_images = cache_images
        self.imgs: list[Image.Image] | None = [] if cache_images else None
        if cache_images:
            for p in self.df["abs_path"].tolist():
                im = Image.open(p)
                im = self._pre(im)
                self.imgs.append(im.copy())
                im.close()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        if self.imgs is None:
            with Image.open(self.df.loc[i, "abs_path"]) as im0:
                im = self._pre(im0).copy()
        else:
            im = self.imgs[i]
        y = torch.from_numpy(self.y_log[i])
        return im, y


class BiomassTiledCached(Dataset):
    def __init__(
        self,
        wide_df: pd.DataFrame,
        *,
        targets: Sequence[str] = TARGETS,
        img_size: int = 512,
        cache_images: bool = True,
    ):
        self.df = wide_df.reset_index(drop=True)
        y = self.df[list(targets)].values.astype(np.float32)
        self.y_log = np.log1p(y)
        self.targets = list(targets)

        self._pre = T.Compose(
            [
                T.Lambda(lambda im: im.convert("RGB")),
                T.Resize((img_size, img_size), antialias=True),
            ]
        )

        self.cache_images = cache_images
        self.tiles: list[tuple[Image.Image, Image.Image]] | None = [] if cache_images else None
        if cache_images:
            for p in self.df["abs_path"].tolist():
                with Image.open(p) as im0:
                    im = im0.convert("RGB")
                    left = im.crop((0, 0, 1000, 1000))
                    right = im.crop((1000, 0, 2000, 1000))
                    left = self._pre(left)
                    right = self._pre(right)
                    self.tiles.append((left.copy(), right.copy()))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        if self.tiles is None:
            with Image.open(self.df.loc[i, "abs_path"]) as im0:
                im = im0.convert("RGB")
                left = im.crop((0, 0, 1000, 1000))
                right = im.crop((1000, 0, 2000, 1000))
                left = self._pre(left)
                right = self._pre(right)
        else:
            left, right = self.tiles[i]
        y = torch.from_numpy(self.y_log[i])
        return left, right, y


class TransformView(Dataset):
    def __init__(self, base: Dataset, tfms):
        self.base = base
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        img, y = self.base[i]
        x = self.tfms(img)
        return x, y


class TiledTransformView(Dataset):
    def __init__(self, base: Dataset, tfms):
        self.base = base
        self.tfms = tfms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        left, right, y = self.base[i]
        x_left = self.tfms(left)
        x_right = self.tfms(right)
        x = torch.stack([x_left, x_right], dim=0)
        return x, y


class TiledTTADataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        tta_n: int = 4,
        bcs_val: float = 0.0,
        hue_val: float = 0.0,
        apply_post_tfms: bool = True,
    ):
        self.base = base
        self.tta_n = int(tta_n)
        if self.tta_n <= 0:
            raise ValueError("tta_n must be >= 1.")
        self.apply_post_tfms = bool(apply_post_tfms)
        self.post = post_tfms() if self.apply_post_tfms else None
        if self.post is None:
            raise ValueError("TiledTTADataset requires apply_post_tfms=True.")
        self.bcs_val = float(bcs_val)
        self.hue_val = float(hue_val)
        self._ops = TTABatch._build_ops(self.tta_n)
        self._jitter = (
            T.ColorJitter(
                brightness=self.bcs_val,
                contrast=self.bcs_val,
                saturation=self.bcs_val,
                hue=self.hue_val,
            )
            if (self.bcs_val != 0.0 or self.hue_val != 0.0)
            else None
        )

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _apply_op(img: Image.Image, k: int, do_hflip: bool) -> Image.Image:
        x_t = img
        if k == 1:
            x_t = x_t.transpose(Image.ROTATE_90)
        elif k == 2:
            x_t = x_t.transpose(Image.ROTATE_180)
        elif k == 3:
            x_t = x_t.transpose(Image.ROTATE_270)
        if do_hflip:
            x_t = x_t.transpose(Image.FLIP_LEFT_RIGHT)
        return x_t

    def __getitem__(self, i: int):
        item = self.base[i]
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            left, right, y = item[0], item[1], item[2]
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            left, right = item[0], item[1]
            y = None
        else:
            raise ValueError("TiledTTADataset expects (left, right[, y]) from base dataset.")

        if torch.is_tensor(left) or torch.is_tensor(right):
            raise ValueError("TiledTTADataset expects PIL images; apply post_tfms inside TiledTTADataset.")

        outs: list[torch.Tensor] = []
        for k, do_hflip in self._ops:
            l = self._apply_op(left, k, do_hflip)
            r = self._apply_op(right, k, do_hflip)
            if self._jitter is not None:
                jitter_fn = T.ColorJitter.get_params(
                    self._jitter.brightness,
                    self._jitter.contrast,
                    self._jitter.saturation,
                    self._jitter.hue,
                )
                l = jitter_fn(l)
                r = jitter_fn(r)
            outs.append(torch.stack([self.post(l), self.post(r)], dim=0))

        x_tta = torch.stack(outs, dim=0)
        if y is None:
            return x_tta
        return x_tta, y


class TTADataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        tta_n: int = 4,
        bcs_val: float = 0.0,
        hue_val: float = 0.0,
        apply_post_tfms: bool = True,
    ):
        self.base = base
        self.tta_n = int(tta_n)
        self.apply_post_tfms = bool(apply_post_tfms)
        self.post = post_tfms() if self.apply_post_tfms else None
        self.tta = TTABatch(tta_n=self.tta_n, bcs_val=float(bcs_val), hue_val=float(hue_val))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        item = self.base[i]
        if isinstance(item, (tuple, list)):
            img, y = item[0], item[1]
        else:
            img, y = item, None

        if torch.is_tensor(img):
            raise ValueError("TTADataset expects PIL images; apply post_tfms inside TTADataset.")
        if self.post is None:
            raise ValueError("TTADataset requires apply_post_tfms=True.")

        tta_imgs = self.tta(img)
        x_tta = torch.stack([self.post(im) for im in tta_imgs], dim=0)
        if y is None:
            return x_tta
        return x_tta, y
