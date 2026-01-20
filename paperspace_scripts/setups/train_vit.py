"""
Minimal ViT pretraining pipeline (self-supervised BYOL-style) + per-epoch validation.

Requirements:
  pip install torch torchvision timm

Usage:
  1) Replace `YourTrainDataset` and `YourValDataset` with your own datasets (no downloading here).
  2) Ensure each dataset returns either:
       - a PIL.Image or torch.Tensor image
       - optionally a label (ignored for self-supervised pretrain)
"""

from __future__ import annotations

import math
import os, tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Optional but convenient for ViT backbones
from torchvision import transforms,datasets
from torchvision import datasets

try:
    import timm  # noqa: F401
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "timm"])
    import timm
    
# -------------------------
# Config
# -------------------------

@dataclass
class Config:
    # Data
    dataset_name: str = "cifar100"
    num_workers: int = 10
    batch_size: int = 128
    image_size: int = 224

    # Train
    epochs: int = 10_000
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip_norm: float = 1.0
    amp: bool = True

    # BYOL / EMA
    ema_base: float = 0.996  # teacher momentum base
    ema_final: float = 1.0   # teacher momentum final
    proj_dim: int = 256
    pred_hidden_dim: int = 4096

    # Checkpoints
    out_dir: str = "./checkpoints_vit_pretrain"
    save_every: int = float("inf")
    save = False

    # Repro
    seed: int = 42
    
    
def log_gpu_stats(prefix: str = "", device: int | None = None) -> dict:

    if not torch.cuda.is_available():
        return {"cuda_available": False}

    if device is None:
        device = torch.cuda.current_device()

    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)

    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_alloc = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    gb = 1024 ** 3
    stats = {
        "cuda_available": True,
        "device": device,
        "name": props.name,
        "total_GB": props.total_memory / gb,
        "allocated_GB": alloc / gb,
        "reserved_GB": reserved / gb,
        "max_allocated_GB": max_alloc / gb,
        "max_reserved_GB": max_reserved / gb,
        "free_est_GB": (props.total_memory - reserved) / gb,  # estimate
    }

    msg = (
        f"{prefix}GPU{device} {stats['name']} | "
        f"alloc {stats['allocated_GB']:.2f} GB | "
        f"reserved {stats['reserved_GB']:.2f} GB | "
        f"max_alloc {stats['max_allocated_GB']:.2f} GB | "
        f"max_res {stats['max_reserved_GB']:.2f} GB | "
        f"free_est {stats['free_est_GB']:.2f} GB"
    )
    print(msg)
    return stats

# -------------------------

# -------------------------


DATASETS: Dict[str, Callable] = {
    "cifar10": lambda root, train, transform: datasets.CIFAR10(
        root=root, train=train, download=True, transform=transform
    ),
    "cifar100": lambda root, train, transform: datasets.CIFAR100(
        root=root, train=train, download=True, transform=transform
    ),
    "mnist": lambda root, train, transform: datasets.MNIST(
        root=root, train=train, download=True, transform=transform
    ),
}


# -------------------------
# Augmentations: produce 2 views per sample
# -------------------------

class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k


def build_ssl_transforms(image_size: int):
    # Works for PIL or torch.Tensor (torchvision transforms support tensors too, for most ops)
    # If your dataset returns PIL Images, you’re good. If it returns Tensors, also fine.
    import torchvision.transforms.functional as TF

    to_tensor = transforms.Lambda(
        lambda x: x if torch.is_tensor(x) else TF.to_tensor(x)  # PIL -> float tensor in [0,1]
    )

    aug = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        to_tensor,
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return TwoCropsTransform(aug)


def ssl_collate_fn(batch):
    """
    Supports dataset items:
      - img
      - (img, label)
    Returns:
      v1, v2 tensors: [B, 3, H, W]
    """
    imgs = []
    for item in batch:
        img = item[0] if isinstance(item, (tuple, list)) else item
        imgs.append(img)

    # If dataset returns tensors with varying sizes, you may need a resize here.
    # We rely on RandomResizedCrop to output the final size.
    v1_list, v2_list = [], []
    for img in imgs:
        v1, v2 = ssl_transform(img)  # uses global ssl_transform
        v1_list.append(v1)
        v2_list.append(v2)

    return torch.stack(v1_list, dim=0), torch.stack(v2_list, dim=0)


# -------------------------
# Model pieces
# -------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOLNet(nn.Module):
    """
    Online network: backbone -> projector -> predictor
    Target network: backbone -> projector (EMA updated)
    """
    def __init__(self, backbone: nn.Module, feat_dim: int, proj_dim: int, pred_hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.projector = _mlp(feat_dim, pred_hidden_dim, proj_dim)
        self.predictor = _mlp(proj_dim, pred_hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)            # [B, feat_dim]
        z = self.projector(feat)           # [B, proj_dim]
        p = self.predictor(z)              # [B, proj_dim]
        return feat, z, p


@torch.no_grad()
def update_ema(teacher: nn.Module, student: nn.Module, m: float):
    # EMA update: teacher = m * teacher + (1-m) * student
    for t_p, s_p in zip(teacher.parameters(), student.parameters()):
        t_p.data.mul_(m).add_(s_p.data, alpha=(1.0 - m))


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Negative cosine similarity between predictor output p and target projection z.
    Both are normalized.
    """
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()


# -------------------------
# Training / Validation
# -------------------------

def cosine_ema_schedule(epoch: int, total_epochs: int, base: float, final: float) -> float:
    # Smoothly increases momentum towards final
    t = epoch / max(1, total_epochs - 1)
    return final - (final - base) * (math.cos(math.pi * t) + 1) / 2


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    *,
    cfg: Config,
    epoch: int,
    student: BYOLNet,
    teacher_backbone: nn.Module,
    teacher_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    student.train()
    teacher_backbone.eval()
    teacher_projector.eval()

    total_loss = 0.0
    n = 0

    m = cosine_ema_schedule(epoch, cfg.epochs, cfg.ema_base, cfg.ema_final)

    start = time.time()
    for step, (v1, v2) in tqdm(enumerate(loader), total=len(loader)):
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=cfg.amp):
            # Online forward
            _, z1, p1 = student(v1)
            _, z2, p2 = student(v2)

            # Teacher forward (no predictor, no grad)
            with torch.no_grad():
                t_feat1 = teacher_backbone(v1)
                t_feat2 = teacher_backbone(v2)
                t_z1 = teacher_projector(t_feat1)
                t_z2 = teacher_projector(t_feat2)

            loss = byol_loss(p1, t_z2) + byol_loss(p2, t_z1)

        if scaler is not None and cfg.amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        # EMA update teacher from student backbone+projector
        with torch.no_grad():
            update_ema(teacher_backbone, student.backbone, m)
            update_ema(teacher_projector, student.projector, m)

        bs = v1.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    dt = time.time() - start
    return {"train_loss": total_loss / max(1, n), "epoch_time_sec": dt, "ema_m": m}


@torch.no_grad()
def validate_one_epoch(
    *,
    cfg: Config,
    student: BYOLNet,
    teacher_backbone: nn.Module,
    teacher_projector: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    student.eval()
    teacher_backbone.eval()
    teacher_projector.eval()

    total_loss = 0.0
    n = 0

    for (v1, v2) in loader:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.amp):
            _, _, p1 = student(v1)
            _, _, p2 = student(v2)

            t_feat1 = teacher_backbone(v1)
            t_feat2 = teacher_backbone(v2)
            t_z1 = teacher_projector(t_feat1)
            t_z2 = teacher_projector(t_feat2)

            loss = byol_loss(p1, t_z2) + byol_loss(p2, t_z1)

        bs = v1.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

    return {"val_loss": total_loss / max(1, n)}


def save_checkpoint(
    path: str,
    *,
    epoch: int,
    student: BYOLNet,
    teacher_backbone: nn.Module,
    teacher_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: Config
):
    if not cfg.save:
        return
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "cfg": cfg.__dict__,
        "student": student.state_dict(),
        "teacher_backbone": teacher_backbone.state_dict(),
        "teacher_projector": teacher_projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
    }
    torch.save(payload, path)


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets (NO downloading)
    # Build datasets (NO downloading)
    if cfg.dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset_name={cfg.dataset_name}. Options: {list(DATASETS.keys())}")

    # SSL transform used inside collate
    ssl_transform = build_ssl_transforms(cfg.image_size)

    # Create datasets using the registry.
    # NOTE: FakeData needs image_size passed; others ignore extra kwargs.
    with tempfile.TemporaryDirectory() as tmp_root:
        TMP_ROOT = os.path.join(tmp_root, "torchvision_data")
        builder = DATASETS[cfg.dataset_name]
        train_ds = builder(TMP_ROOT, True,  None)
        val_ds   = builder(TMP_ROOT, False, None)

    # SSL transform used inside collate
    ssl_transform = build_ssl_transforms(cfg.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ssl_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=ssl_collate_fn,
    )

    # Build ViT backbone: output features (no classifier head)
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,          # returns feature vector
        global_pool="avg",      # stable default; can also use "token"
    )

    # Figure out feature dimension
    # timm models usually expose num_features
    feat_dim = getattr(backbone, "num_features", None)
    if feat_dim is None:
        raise RuntimeError("Could not infer backbone feature dim; check your timm model.")

    student = BYOLNet(
        backbone=backbone,
        feat_dim=feat_dim,
        proj_dim=cfg.proj_dim,
        pred_hidden_dim=cfg.pred_hidden_dim,
    ).to(device)

    # Teacher modules start as a copy of student's backbone+projector
    teacher_backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    ).to(device)
    teacher_projector = _mlp(feat_dim, cfg.pred_hidden_dim, cfg.proj_dim).to(device)

    teacher_backbone.load_state_dict(student.backbone.state_dict())
    teacher_projector.load_state_dict(student.projector.state_dict())

    for p in teacher_backbone.parameters():
        p.requires_grad = False
    for p in teacher_projector.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        tr = train_one_epoch(
            cfg=cfg,
            epoch=epoch,
            student=student,
            teacher_backbone=teacher_backbone,
            teacher_projector=teacher_projector,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
        )
        va = validate_one_epoch(
            cfg=cfg,
            student=student,
            teacher_backbone=teacher_backbone,
            teacher_projector=teacher_projector,
            loader=val_loader,
            device=device,
        )

        msg = (
            f"Epoch {epoch+1:03d}/{cfg.epochs} | "
            f"train_loss={tr['train_loss']:.4f} | val_loss={va['val_loss']:.4f} | "
            f"ema_m={tr['ema_m']:.5f} | time={tr['epoch_time_sec']:.1f}s"
        )
        print(msg)

        # Save best + periodic
        if va["val_loss"] < best_val:
            best_val = va["val_loss"]
            save_checkpoint(
                os.path.join(cfg.out_dir, "best.pt"),
                epoch=epoch,
                student=student,
                teacher_backbone=teacher_backbone,
                teacher_projector=teacher_projector,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
            )

        if (epoch + 1) % cfg.save_every == 0:
            save_checkpoint(
                os.path.join(cfg.out_dir, f"epoch_{epoch+1:03d}.pt"),
                epoch=epoch,
                student=student,
                teacher_backbone=teacher_backbone,
                teacher_projector=teacher_projector,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
            )
        log_gpu_stats()

    print(f"Done. Best val_loss: {best_val:.4f}. Checkpoints in: {cfg.out_dir}")
