from __future__ import annotations

from typing import Any

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


class DINOv3Regressor(nn.Module):
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
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_dtype = backbone_dtype
        self.backbone_grad = False
        self.pred_space = _normalize_pred_space(pred_space)

        feat_dim = feat_dim or getattr(getattr(backbone, "norm", None), "normalized_shape", [None])[0]
        if feat_dim is None:
            raise ValueError("Could not infer feat_dim from backbone; pass feat_dim=... explicitly.")

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.feat_dim = int(feat_dim)
        norm_layer = nn.LayerNorm if norm_layer is None else norm_layer

        self.neck: nn.ModuleList
        if int(num_neck) > 0:
            SelfAttentionBlock = _optional_import_self_attention_block()
            self.neck = nn.ModuleList(
                [SelfAttentionBlock(self.feat_dim, num_heads=int(neck_num_heads)) for _ in range(int(num_neck))]
            )
        else:
            self.neck = nn.ModuleList()

        self.head = _build_head(
            in_dim=self.feat_dim,
            hidden=int(hidden),
            depth=int(depth),
            drop=float(drop),
            out_dim=int(out_dim),
            norm_layer=norm_layer,
        )
        self.norm = norm_layer(self.feat_dim)

    @torch.no_grad()
    def _backbone_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
        out = self.backbone(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone_grad):
            with autocast_context(x.device, dtype=self.backbone_dtype):
                tokens, rope = self._backbone_tokens(x)

        for block in self.neck:
            try:
                tokens = block(tokens, rope)
            except TypeError:
                tokens = block(tokens)

        cls = tokens[:, 0, :]
        cls = self.norm(cls)
        return self.head(cls)

    def set_train(self, train: bool = True) -> None:
        self.neck.train(train)
        self.head.train(train)
        self.norm.train(train)

    def set_backbone_grad(self, train: bool = True) -> None:
        self.backbone_grad = bool(train)

    @torch.no_grad()
    def init(self) -> None:
        modules = [*self.head.modules(), *self.neck.modules(), *self.norm.modules()]
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, "elementwise_affine", False):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


class DINOv3Regressor3(DINOv3Regressor):
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
                in_dim=self.feat_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head_clover = _build_head(
                in_dim=self.feat_dim,
                hidden=int(hidden),
                depth=int(depth),
                drop=float(drop),
                out_dim=1,
                norm_layer=norm_layer,
            )
            self.head_dead = _build_head(
                in_dim=self.feat_dim,
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
            gdm = torch.log1p(gdm_lin)
            total = torch.log1p(total_lin)
            return torch.cat([green, clover, dead, gdm, total], dim=1)
        gdm = green + clover
        total = gdm + dead
        return torch.cat([green, clover, dead, gdm, total], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone_grad):
            with autocast_context(x.device, dtype=self.backbone_dtype):
                tokens, rope = self._backbone_tokens(x)

        for block in self.neck:
            try:
                tokens = block(tokens, rope)
            except TypeError:
                tokens = block(tokens)

        cls = tokens[:, 0, :]
        cls = self.norm(cls)
        green, clover, dead = self._split_components(cls)
        return self._compose_outputs(green, clover, dead)

    def set_train(self, train: bool = True) -> None:
        self.neck.train(train)
        self.norm.train(train)
        if self.head_style == "multi":
            self.head_green.train(train)
            self.head_clover.train(train)
            self.head_dead.train(train)
        else:
            self.head.train(train)

    @torch.no_grad()
    def init(self) -> None:
        modules = [*self.neck.modules(), *self.norm.modules()]
        if self.head_style == "multi":
            modules += [*self.head_green.modules(), *self.head_clover.modules(), *self.head_dead.modules()]
        else:
            modules += [*self.head.modules()]
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, "elementwise_affine", False):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


class TiledDINOv3Regressor(DINOv3Regressor):
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
    ):
        norm_layer = nn.LayerNorm if norm_layer is None else norm_layer
        super().__init__(
            backbone,
            hidden=int(hidden),
            depth=int(depth),
            drop=float(drop),
            out_dim=int(out_dim),
            feat_dim=feat_dim,
            norm_layer=norm_layer,
            num_neck=int(num_neck),
            neck_num_heads=int(neck_num_heads),
            backbone_dtype=backbone_dtype,
            pred_space=pred_space,
        )
        self.fused_dim = int(self.feat_dim) * 2
        self.head = _build_head(
            in_dim=self.fused_dim,
            hidden=int(hidden),
            depth=int(depth),
            drop=float(drop),
            out_dim=int(out_dim),
            norm_layer=norm_layer,
        )

    def _tile_cls(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.backbone_grad):
            with autocast_context(x.device, dtype=self.backbone_dtype):
                tokens, rope = self._backbone_tokens(x)

        for block in self.neck:
            try:
                tokens = block(tokens, rope)
            except TypeError:
                tokens = block(tokens)

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

    def set_train(self, train: bool = True) -> None:
        self.neck.train(train)
        self.norm.train(train)
        if self.head_style == "multi":
            self.head_green.train(train)
            self.head_clover.train(train)
            self.head_dead.train(train)
        else:
            self.head.train(train)

    @torch.no_grad()
    def init(self) -> None:
        modules = [*self.neck.modules(), *self.norm.modules()]
        if self.head_style == "multi":
            modules += [*self.head_green.modules(), *self.head_clover.modules(), *self.head_dead.modules()]
        else:
            modules += [*self.head.modules()]
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if getattr(m, "elementwise_affine", False):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


def _optional_import_self_attention_block():
    try:
        from dinov3.layers.block import SelfAttentionBlock  # type: ignore

        return SelfAttentionBlock
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "num_neck>0 requires dinov3 on PYTHONPATH (e.g. add your dinov3 repo to sys.path)."
        ) from e
