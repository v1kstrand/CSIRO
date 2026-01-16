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


def _clean_image(img: Image.Image) -> Image.Image:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    arr = arr[: int(h * 0.90), :]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if int(mask.sum()) > 0:
        arr = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(arr)


def _maybe_preprocess_image(img: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return img
    return _clean_image(img)


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
        img_preprocess: bool = False,
        pad_to_square: bool = True,
        pad_fill: int = 0,
    ):
        self.df = wide_df.reset_index(drop=True)
        y = self.df[list(targets)].values.astype(np.float32)
        self.y_log = np.log1p(y)
        self.targets = list(targets)
        self.img_preprocess = bool(img_preprocess)

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
                im = Image.open(p).convert("RGB")
                im = _maybe_preprocess_image(im, self.img_preprocess)
                im = self._pre(im)
                self.imgs.append(im.copy())
                im.close()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        if self.imgs is None:
            with Image.open(self.df.loc[i, "abs_path"]) as im0:
                im = im0.convert("RGB")
                im = _maybe_preprocess_image(im, self.img_preprocess)
                im = self._pre(im).copy()
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
        img_preprocess: bool = False,
    ):
        self.df = wide_df.reset_index(drop=True)
        y = self.df[list(targets)].values.astype(np.float32)
        self.y_log = np.log1p(y)
        self.targets = list(targets)
        self.img_preprocess = bool(img_preprocess)

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
                    im = _maybe_preprocess_image(im, self.img_preprocess)
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
                im = _maybe_preprocess_image(im, self.img_preprocess)
                left = im.crop((0, 0, 1000, 1000))
                right = im.crop((1000, 0, 2000, 1000))
                left = self._pre(left)
                right = self._pre(right)
        else:
            left, right = self.tiles[i]
        y = torch.from_numpy(self.y_log[i])
        return left, right, y


class BiomassFullCached(Dataset):
    def __init__(
        self,
        wide_df: pd.DataFrame,
        *,
        targets: Sequence[str] = TARGETS,
        cache_images: bool = True,
        img_preprocess: bool = False,
    ):
        self.df = wide_df.reset_index(drop=True)
        y = self.df[list(targets)].values.astype(np.float32)
        self.y_log = np.log1p(y)
        self.targets = list(targets)
        self.img_preprocess = bool(img_preprocess)
        self.cache_images = cache_images
        self.imgs: list[Image.Image] | None = [] if cache_images else None
        if cache_images:
            for p in self.df["abs_path"].tolist():
                im = Image.open(p).convert("RGB")
                im = _maybe_preprocess_image(im, self.img_preprocess)
                self.imgs.append(im.copy())
                im.close()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        if self.imgs is None:
            with Image.open(self.df.loc[i, "abs_path"]) as im0:
                im = im0.convert("RGB")
                im = _maybe_preprocess_image(im, self.img_preprocess)
                im = im.copy()
        else:
            im = self.imgs[i]
        y = torch.from_numpy(self.y_log[i])
        return im, y


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


class TiledSharedTransformView(Dataset):
    def __init__(self, base: Dataset, geom_tfms, *, img_size: int = 512, post=None):
        self.base = base
        self.geom_tfms = geom_tfms
        self.resize = T.Resize((int(img_size), int(img_size)), antialias=True)
        self.post = post if post is not None else post_tfms()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        item = self.base[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            img, y = item[0], item[1]
        else:
            img, y = item, None

        if not isinstance(img, Image.Image):
            raise ValueError("TiledSharedTransformView expects PIL images from base dataset.")

        if self.geom_tfms is not None:
            if isinstance(self.geom_tfms, str) and self.geom_tfms == "safe":
                k = int(torch.randint(0, 4, (1,)).item())
                if k == 1:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                elif k == 2:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif k == 3:
                    img = img.transpose(Image.ROTATE_180)
            else:
                img = self.geom_tfms(img)

        left = img.crop((0, 0, 1000, 1000))
        right = img.crop((1000, 0, 2000, 1000))

        left = self.post(self.resize(left))
        right = self.post(self.resize(right))
        x = torch.stack([left, right], dim=0)
        if y is None:
            return x
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


class TiledSharedTTADataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        tta_n: int = 4,
        bcs_val: float = 0.0,
        hue_val: float = 0.0,
        img_size: int = 512,
        apply_post_tfms: bool = True,
    ):
        self.base = base
        self.tta_n = int(tta_n)
        if self.tta_n <= 0:
            raise ValueError("tta_n must be >= 1.")
        self.apply_post_tfms = bool(apply_post_tfms)
        self.post = post_tfms() if self.apply_post_tfms else None
        if self.post is None:
            raise ValueError("TiledSharedTTADataset requires apply_post_tfms=True.")
        self.resize = T.Resize((int(img_size), int(img_size)), antialias=True)
        self.bcs_val = float(bcs_val)
        self.hue_val = float(hue_val)
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
        self._ops = list(range(4))

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _apply_op(img: Image.Image, k: int) -> Image.Image:
        if k == 1:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        if k == 2:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        if k == 3:
            return img.transpose(Image.ROTATE_180)
        return img

    def __getitem__(self, i: int):
        item = self.base[i]
        if isinstance(item, (tuple, list)):
            img, y = item[0], item[1]
        else:
            img, y = item, None

        if not isinstance(img, Image.Image):
            raise ValueError("TiledSharedTTADataset expects PIL images from base dataset.")

        outs: list[torch.Tensor] = []
        for idx in range(int(self.tta_n)):
            k = self._ops[idx % 4]
            full = self._apply_op(img, k)
            left = full.crop((0, 0, 1000, 1000))
            right = full.crop((1000, 0, 2000, 1000))
            if self._jitter is not None:
                params = T.ColorJitter.get_params(
                    self._jitter.brightness,
                    self._jitter.contrast,
                    self._jitter.saturation,
                    self._jitter.hue,
                )
                params = [float(p) for p in params]
                left = T.functional.adjust_brightness(left, params[0])
                left = T.functional.adjust_contrast(left, params[1])
                left = T.functional.adjust_saturation(left, params[2])
                left = T.functional.adjust_hue(left, params[3])
                right = T.functional.adjust_brightness(right, params[0])
                right = T.functional.adjust_contrast(right, params[1])
                right = T.functional.adjust_saturation(right, params[2])
                right = T.functional.adjust_hue(right, params[3])
            left = self.post(self.resize(left))
            right = self.post(self.resize(right))
            outs.append(torch.stack([left, right], dim=0))

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
