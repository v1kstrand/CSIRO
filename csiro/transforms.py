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
    ):
        self.tta_n = int(tta_n)
        if self.tta_n <= 0:
            raise ValueError("tta_n must be >= 1.")
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
        self._ops = self._build_ops(self.tta_n)

    @staticmethod
    def _build_ops(tta_n: int) -> list[tuple[int, bool]]:
        base = [(k, False) for k in range(4)] + [(k, True) for k in range(4)]
        if tta_n <= 8:
            return base[:tta_n]
        return [base[i % 8] for i in range(tta_n)]

    def __call__(self, img: Image.Image) -> list[Image.Image]:
        if torch.is_tensor(img):
            raise ValueError("TTABatch expects PIL images; apply post_tfms after TTA.")
        if not isinstance(img, Image.Image):
            raise ValueError(f"TTABatch expects PIL images, got {type(img)}.")

        outs: list[Image.Image] = []
        for k, do_hflip in self._ops:
            x_t = img
            if k == 1:
                x_t = x_t.transpose(Image.ROTATE_90)
            elif k == 2:
                x_t = x_t.transpose(Image.ROTATE_180)
            elif k == 3:
                x_t = x_t.transpose(Image.ROTATE_270)
            if do_hflip:
                x_t = x_t.transpose(Image.FLIP_LEFT_RIGHT)
            if self._jitter is not None:
                x_t = self._jitter(x_t)
            outs.append(x_t)
        return outs
