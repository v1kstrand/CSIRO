from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn as nn

from .amp import autocast_context

def _normalize_pred_space(pred_space: str) -> str:
    s = str(pred_space).strip().lower()
    if s in ("log", "log1p"):
        return "log"
    if s in ("gram", "grams", "linear"):
        return "gram"
    raise ValueError(f"Unknown pred_space: {pred_space}")

def _build_head(
    *,
    in_dim: int,
    hidden: int,
    depth: int,
    drop: float,
    out_dim: int,
    norm_layer: type[nn.Module],
) -> nn.Sequential:
    if depth < 2:
        raise ValueError(f"depth must be >= 2 (got {depth})")
    layers: list[nn.Module] = []
    d = int(in_dim)
    for _ in range(int(depth) - 1):
        layers += [nn.Linear(d, hidden), norm_layer(hidden), nn.GELU(), nn.Dropout(drop)]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def _infer_feat_dim(backbone: nn.Module) -> int:
    feat_dim = getattr(getattr(backbone, "norm", None), "normalized_shape", [None])[0]
    if feat_dim is None:
        raise ValueError("Could not infer feat_dim from backbone; pass feat_dim=... explicitly.")
    return int(feat_dim)


@torch.no_grad()
def _backbone_tokens(backbone: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
    out = backbone(x)
    rope = None
    if isinstance(out, tuple) and len(out) == 2:
        out, rope = out

    tokens = out
    if isinstance(out, dict):
        if "x_prenorm" in out:
            tokens = out["x_prenorm"]
        elif "x_norm_clstoken" in out:
            cls = out["x_norm_clstoken"]
            if cls.ndim == 2:
                cls = cls[:, None, :]
            patch = out.get("x_norm_patchtokens", None)
            if patch is not None:
                tokens = torch.cat([cls, patch], dim=1)
            else:
                tokens = cls
        else:
            for k in ("x_norm_clstoken", "cls", "cls_token", "clstoken", "x"):
                if k in out:
                    tokens = out[k]
                    break
            else:
                raise ValueError(f"Backbone returned dict with unknown keys: {list(out.keys())}")

    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Backbone output must be a Tensor/dict/tuple, got: {type(out)!r}")

    if tokens.ndim == 2:
        tokens = tokens[:, None, :]
    if tokens.ndim != 3:
        raise ValueError(f"Expected tokens [B,T,D], got shape: {tuple(tokens.shape)}")
    return tokens, rope


def _init_modules(modules: list[nn.Module]) -> None:
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if getattr(m, "elementwise_affine", False):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class TiledDINOv3Regressor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden: int = 1024,
        depth: int = 2,
        drop: float = 0.1,
        out_dim: int = 5,
        feat_dim: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        num_neck: int = 0,
        neck_num_heads: int = 12,
        backbone_dtype: torch.dtype | None = None,
        pred_space: str = "log",
        neck_rope: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_dtype = backbone_dtype
        self.backbone_grad = False
        self.pred_space = _normalize_pred_space(pred_space)
        self.neck_rope = bool(neck_rope)

        feat_dim = feat_dim or _infer_feat_dim(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.feat_dim = int(feat_dim)
        norm_layer = nn.LayerNorm if norm_layer is None else norm_layer

        if int(num_neck) > 0:
            SelfAttentionBlock = _optional_import_self_attention_block()
            self.neck = nn.ModuleList(
                [SelfAttentionBlock(self.feat_dim, num_heads=int(neck_num_heads)) for _ in range(int(num_neck))]
            )
        else:
            self.neck = nn.ModuleList()

        self.fused_dim = int(self.feat_dim) * 2
        self.head = _build_head(
            in_dim=self.fused_dim,
            hidden=int(hidden),
            depth=int(depth),
            drop=float(drop),
            out_dim=int(out_dim),
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(self.feat_dim)

    def _tile_cls(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone_grad):
            with autocast_context(x.device, dtype=self.backbone_dtype):
                tokens, rope = _backbone_tokens(self.backbone, x)

        rope_use = rope if self.neck_rope else None
        for block in self.neck:
            try:
                tokens = block(tokens, rope_use)
            except TypeError:
                if rope_use is None:
                    tokens = block(tokens)
                else:
                    assert False, "SelfAttentionBlock requires rope"

        cls = tokens[:, 0, :]
        return self.norm(cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.size(1) != 2:
            raise ValueError(f"Expected tiled input [B,2,C,H,W], got {tuple(x.shape)}")
        x_left = x[:, 0]
        x_right = x[:, 1]
        cls_left = self._tile_cls(x_left)
        cls_right = self._tile_cls(x_right)
        fused = torch.cat([cls_left, cls_right], dim=1)
        return self.head(fused)

    @torch.no_grad()
    def init(self) -> None:
        modules = [*self.head.modules(), *self.neck.modules(), *self.norm.modules()]
        _init_modules(modules)


class TiledDINOv3Regressor3(TiledDINOv3Regressor):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden: int = 1024,
        depth: int = 2,
        drop: float = 0.1,
        feat_dim: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        num_neck: int = 0,
        neck_num_heads: int = 12,
        backbone_dtype: torch.dtype | None = None,
        pred_space: str = "log",
        head_style: str = "single",
    ):
        head_style = str(head_style).strip().lower()
        if head_style not in ("single", "multi"):
            raise ValueError(f"head_style must be 'single' or 'multi' (got {head_style})")
        norm_layer = nn.LayerNorm if norm_layer is None else norm_layer
        super().__init__(
            backbone,
            hidden=int(hidden),
            depth=int(depth),
            drop=float(drop),
            out_dim=3,
            feat_dim=feat_dim,
            norm_layer=norm_layer,
            num_neck=int(num_neck),
            neck_num_heads=int(neck_num_heads),
            backbone_dtype=backbone_dtype,
            pred_space=pred_space,
        )
        self.head_style = head_style
        if head_style == "multi":
            self.head_green = _build_head(
                in_dim=self.fused_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head_clover = _build_head(
                in_dim=self.fused_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head_dead = _build_head(
                in_dim=self.fused_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head = nn.ModuleList([self.head_green, self.head_clover, self.head_dead])

    def _split_components(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.head_style == "multi":
            green = self.head_green(feats)
            clover = self.head_clover(feats)
            dead = self.head_dead(feats)
        else:
            out = self.head(feats)
            green = out[:, [0]]
            clover = out[:, [1]]
            dead = out[:, [2]]
        return green, clover, dead

    def _compose_outputs(self, green: torch.Tensor, clover: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
        if self.pred_space == "log":
            green_lin = torch.expm1(green)
            clover_lin = torch.expm1(clover)
            dead_lin = torch.expm1(dead)
            gdm_lin = green_lin + clover_lin
            total_lin = gdm_lin + dead_lin
            gdm_lin = gdm_lin.clamp_min(-0.999999)
            total_lin = total_lin.clamp_min(-0.999999)
            gdm = torch.log1p(gdm_lin)
            total = torch.log1p(total_lin)
            return torch.cat([green, clover, dead, gdm, total], dim=1)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, clover, dead, gdm, total], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.size(1) != 2:
            raise ValueError(f"Expected tiled input [B,2,C,H,W], got {tuple(x.shape)}")
        x_left = x[:, 0]
        x_right = x[:, 1]
        cls_left = self._tile_cls(x_left)
        cls_right = self._tile_cls(x_right)
        fused = torch.cat([cls_left, cls_right], dim=1)
        green, clover, dead = self._split_components(fused)
        return self._compose_outputs(green, clover, dead)

    @torch.no_grad()
    def init(self) -> None:
        modules = [*self.neck.modules(), *self.norm.modules()]
        if self.head_style == "multi":
            modules += [*self.head_green.modules(), *self.head_clover.modules(), *self.head_dead.modules()]
        else:
            modules += [*self.head.modules()]
        _init_modules(modules)


class TiledDINOv3RegressorStitched3(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden: int = 1024,
        depth: int = 2,
        drop: float = 0.1,
        feat_dim: int | None = None,
        norm_layer: type[nn.Module] | None = None,
        num_neck: int = 0,
        neck_num_heads: int = 12,
        backbone_dtype: torch.dtype | None = None,
        pred_space: str = "log",
        head_style: str = "single",
        out_format: str = "cat_cls",
        neck_rope: bool = True,
        neck_drop: float = 0.0,
        drop_path: dict[str, float] | None = None,
        
    ):
        super().__init__()
        head_style = str(head_style).strip().lower()
        if head_style not in ("single", "multi"):
            raise ValueError(f"head_style must be 'single' or 'multi' (got {head_style})")
        self.backbone = backbone
        self.backbone_dtype = backbone_dtype
        self.backbone_grad = False
        self.pred_space = _normalize_pred_space(pred_space)
        self.out_format = str(out_format).strip().lower()
        self.neck_rope = bool(neck_rope)
        self.num_regs = backbone.n_storage_tokens + 1 
        neck_drop = float(neck_drop)
        if neck_drop < 0.0 or neck_drop > 1.0:
            raise ValueError(f"neck_drop must be in [0,1] (got {neck_drop}).")
        if drop_path is not None:
            assert isinstance(drop_path, dict) and "backbone" in drop_path and "neck" in drop_path
            assert 0.0 <= drop_path["backbone"] <= 1.0 and 0.0 <= drop_path["neck"] <= 1.0
        drop_path = drop_path or {"backbone" : 0.0, "neck" : 0.0}

        feat_dim = feat_dim or _infer_feat_dim(backbone)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if drop_path is not None and drop_path["backbone"] > 0.0:
            for i in range(len(self.backbone.blocks)):
                self.backbone.blocks[i].sample_drop_ratio = drop_path["backbone"]

        self.feat_dim = int(feat_dim)
        norm_layer = nn.LayerNorm if norm_layer is None else norm_layer
        if self.out_format == "mean":
            self.out_dim = self.feat_dim
        elif self.out_format == "cat_cls":
            self.out_dim = self.feat_dim * 2
        elif self.out_format == "cat_cls_w_mean":
            self.out_dim = self.feat_dim * 3
        else:
            raise ValueError(f"Unknown out_format: {self.out_format}")

        if int(num_neck) > 0:
            SelfAttentionBlock = _optional_import_self_attention_block()
            self.neck = nn.ModuleList(
                [
                    SelfAttentionBlock(
                        self.feat_dim,
                        num_heads=int(neck_num_heads),
                        drop=float(neck_drop),
                        attn_drop=float(neck_drop),
                    )
                    for _ in range(int(num_neck))
                ]
            )
        else:
            self.neck = nn.ModuleList()
        
        neck_drop_path = drop_path["neck"] if drop_path is not None else 0.0
        for neck in self.neck:
            neck.sample_drop_ratio = neck_drop_path

        self.norm_bb = norm_layer(self.feat_dim)
        self.norm_neck = norm_layer(self.feat_dim) if int(num_neck) > 0 else nn.Identity()
        self.head_style = head_style
        if head_style == "multi":
            self.head_green = _build_head(
                in_dim=self.out_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head_clover = _build_head(
                in_dim=self.out_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head_dead = _build_head(
                in_dim=self.out_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head = nn.ModuleList([self.head_green, self.head_clover, self.head_dead])
        else:
            self.head = _build_head(
                in_dim=self.out_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=3,
                norm_layer=norm_layer,
            )

    def _split_components(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.head_style == "multi":
            green = self.head_green(feats)
            clover = self.head_clover(feats)
            dead = self.head_dead(feats)
        else:
            out = self.head(feats)
            green = out[:, [0]]
            clover = out[:, [1]]
            dead = out[:, [2]]
        return green, clover, dead

    def _compose_outputs(self, green: torch.Tensor, clover: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
        if self.pred_space == "log":
            green_lin = torch.expm1(green)
            clover_lin = torch.expm1(clover)
            dead_lin = torch.expm1(dead)
            gdm_lin = green_lin + clover_lin
            total_lin = gdm_lin + dead_lin
            gdm_lin = gdm_lin.clamp_min(-0.999999)
            total_lin = total_lin.clamp_min(-0.999999)
            gdm = torch.log1p(gdm_lin)
            total = torch.log1p(total_lin)
            return torch.cat([green, clover, dead, gdm, total], dim=1)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, clover, dead, gdm, total], dim=1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.size(1) != 2:
            raise ValueError(f"Expected tiled input [B,2,C,H,W], got {tuple(x.shape)}")
        x_left = x[:, 0]
        x_right = x[:, 1]
        with torch.set_grad_enabled(self.backbone_grad):
            with autocast_context(x.device, dtype=self.backbone_dtype):
                tok1, rope = _backbone_tokens(self.backbone, x_left)
                tok2, _ = _backbone_tokens(self.backbone, x_right)
        tok1 = self.norm_bb(tok1)
        tok2 = self.norm_bb(tok2)

        if tok1.size(1) <= int(self.num_regs):
            raise ValueError(f"Unexpected token length {tok1.size(1)} for num_regs={self.num_regs}.")
        if tok2.size(1) <= int(self.num_regs):
            raise ValueError(f"Unexpected token length {tok2.size(1)} for num_regs={self.num_regs}.")
        prefix1, patch1 = tok1[:, : int(self.num_regs), :], tok1[:, int(self.num_regs) :, :]
        prefix2, patch2 = tok2[:, : int(self.num_regs), :], tok2[:, int(self.num_regs) :, :]
        cls1, regs1 = prefix1[:, :1, :], prefix1[:, 1:, :]
        cls2, regs2 = prefix2[:, :1, :], prefix2[:, 1:, :]
        bsz, n_tok, dim = patch1.shape
        p = int(math.sqrt(int(n_tok)))
        if p * p != int(n_tok):
            raise ValueError(f"Patch tokens must form a square grid (got {n_tok}).")
        if p % 2 != 0:
            raise ValueError(f"Grid size P must be even for 1x2 pooling (got {p}).")
        
        grid1 = patch1.reshape(bsz, p, p, dim)
        grid2 = patch2.reshape(bsz, p, p, dim)
        grid1_half = grid1.reshape(bsz, p, p // 2, 2, dim).mean(dim=3)
        grid2_half = grid2.reshape(bsz, p, p // 2, 2, dim).mean(dim=3)
        grid_full = torch.cat([grid1_half, grid2_half], dim=2)
        tok_grid = grid_full.reshape(bsz, p * p, dim)

        tokens = torch.cat([cls1, cls2, regs1, regs2, tok_grid], dim=1)
        rope_neck = rope if self.neck_rope else None
        for block in self.neck:
            try:
                tokens = block(tokens, rope_neck)
            except TypeError:
                if rope_neck is None:
                    tokens = block(tokens)
                else:
                    assert False, "SelfAttentionBlock requires rope"
        tokens = self.norm_neck(tokens)
        
        if self.out_format == "mean":
            return tokens.mean(dim=1)
        if self.out_format == "cat_cls":
            cls1, cls2 = tokens[:, 0, :], tokens[:, 1, :]
            return torch.cat([cls1, cls2], dim=1)
        if self.out_format == "cat_cls_w_mean":
            cls1, cls2 = tokens[:, 0, :], tokens[:, 1, :]
            mean = tokens[:, 2:, :].mean(dim=1)
            return torch.cat([cls1, cls2, mean], dim=1)
        raise ValueError(f"Unknown out_format: {self.out_format}")



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode(x)
        green, clover, dead = self._split_components(feats)
        return self._compose_outputs(green, clover, dead)

    def set_backbone_grad(self, train: bool = True) -> None:
        self.backbone_grad = bool(train)

    @torch.no_grad()
    def init(self) -> None:
        modules = [*self.neck.modules(), *self.norm_bb.modules(), *self.norm_neck.modules()]
        if self.head_style == "multi":
            modules += [*self.head_green.modules(), *self.head_clover.modules(), *self.head_dead.modules()]
        else:
            modules += [*self.head.modules()]
        _init_modules(modules)


def _optional_import_self_attention_block():
    try:
        from dinov3.layers.block import SelfAttentionBlock  # type: ignore

        return SelfAttentionBlock
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "num_neck>0 requires dinov3 on PYTHONPATH (e.g. add your dinov3 repo to sys.path)."
        ) from e
