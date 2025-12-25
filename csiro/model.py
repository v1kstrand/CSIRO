from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .amp import autocast_context

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
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_dtype = backbone_dtype

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

        if depth < 2:
            raise ValueError(f"depth must be >= 2 (got {depth})")

        layers: list[nn.Module] = []
        in_dim = self.feat_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), norm_layer(hidden), nn.GELU(), nn.Dropout(drop)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, out_dim)]
        self.head = nn.Sequential(*layers)
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
        with torch.no_grad():
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


def _optional_import_self_attention_block():
    try:
        from dinov3.layers.block import SelfAttentionBlock  # type: ignore

        return SelfAttentionBlock
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "num_neck>0 requires dinov3 on PYTHONPATH (e.g. add your dinov3 repo to sys.path)."
        ) from e
