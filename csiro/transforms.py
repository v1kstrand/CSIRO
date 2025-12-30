from __future__ import annotations

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch

from .config import IMAGENET_MEAN, IMAGENET_STD


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        new = Image.new(img.mode, (s, s), color=self.fill)
        new.paste(img, ((s - w) // 2, (s - h) // 2))
        return new


base_train_comp = T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomChoice(
            [
                T.Lambda(lambda x: x),
                T.Lambda(lambda x: x.transpose(Image.ROTATE_90)),
                T.Lambda(lambda x: x.transpose(Image.ROTATE_180)),
                T.Lambda(lambda x: x.transpose(Image.ROTATE_270)),
            ]
        ),
    ]
)

def train_tfms():
    return T.Compose(
        [
            base_train_comp,
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.035),
        ]
    )
    



def post_tfms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

train_tfms_dict = {
    "default": train_tfms,
}


class TTABatch:
    def __init__(
        self,
        *,
        tta_n: int = 4,
        bcs_val: float = 0.0,
        hue_val: float = 0.0,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ):
        self.tta_n = int(tta_n)
        if self.tta_n <= 0:
            raise ValueError("tta_n must be >= 1.")
        self.bcs_val = float(bcs_val)
        self.hue_val = float(hue_val)
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std = torch.tensor(std).view(1, 3, 1, 1)
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
        self._ops = self._build_ops(self.tta_n)

    @staticmethod
    def _build_ops(tta_n: int) -> list[tuple[int, bool]]:
        base = [(k, False) for k in range(4)] + [(k, True) for k in range(4)]
        if tta_n <= 8:
            return base[:tta_n]
        return [base[i % 8] for i in range(tta_n)]

    def _unnorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _apply_jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.bcs_val == 0.0 and self.hue_val == 0.0:
            return x
        x = self._unnorm(x).clamp(0.0, 1.0)
        x = self._jitter(x)
        x = x.clamp(0.0, 1.0)
        return self._norm(x)

    def __call__(self, x: torch.Tensor, *, flatten: bool = True) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected input [B,C,H,W] or [C,H,W], got {tuple(x.shape)}")

        outs: list[torch.Tensor] = []
        for i, (k, do_hflip) in enumerate(self._ops):
            x_t = x if k == 0 else torch.rot90(x, k, dims=(-2, -1))
            if do_hflip:
                x_t = torch.flip(x_t, dims=(-1,))
            x_t = self._apply_jitter(x_t)
            outs.append(x_t)

        out = torch.stack(outs, dim=1)
        if flatten:
            b, t, c, h, w = out.shape
            return out.view(b * t, c, h, w)
        return out
