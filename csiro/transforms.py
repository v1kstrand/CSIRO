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


def get_train_tfms():
    return T.Compose(
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
            ),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.035),
        ]
    )


def get_post_tfms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])


def get_post_tfms_imagenet():
    return get_post_tfms(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# Notebook-friendly aliases
get_tfms = get_train_tfms


def post_tfms():
    return get_post_tfms_imagenet()
