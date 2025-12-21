from __future__ import annotations

from PIL import Image
import torchvision.transforms as T

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
                    T.RandomRotation((90, 90)),
                    T.RandomRotation((180, 180)),
                    T.RandomRotation((270, 270)),
                ]
            ),])

def train_tfms():
    return T.Compose(
        [
            base_train_comp,
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.035),
        ]
    )
    
def train_tfms_v1():
    return T.Compose(
        [
            base_train_comp,
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ]
    )
    
def train_tfms_v2():
    return T.Compose(
        [
            base_train_comp,
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        ]
    )
    
def train_tfms_v3():
    return T.Compose(
        [
            base_train_comp,
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
        ]
    )
    
def train_tfms_v4():
    return T.Compose(
        [
            base_train_comp,
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ]
    )
    
def train_tfms_v5():
    return T.Compose(
        [
            base_train_comp,
        ]
    )

def post_tfms(mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

train_tfms_dict = {
    "default": train_tfms,
    "v1": train_tfms_v1,
    "v2": train_tfms_v2,
    "v3": train_tfms_v3,
    "v4": train_tfms_v4,
    "v5": train_tfms_v5
}