import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _to_bchw(latent: torch.Tensor, hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Supports two input formats:
    - (B, C, H, W): returned as-is
    - (B, N, C): infer/use hw and reshape tokens to (B, C, H, W)
    """
    if latent.dim() == 4:
        return latent

    if latent.dim() != 3:
        raise ValueError(f"Expected latent dim 3 or 4, got {latent.shape}")

    b, n, c = latent.shape
    if hw is not None:
        h, w = hw
        if h * w != n:
            raise ValueError(f"token_hw={hw} but N={n} mismatch (H*W must equal N).")
    else:
        s = int(math.sqrt(n))
        if s * s != n:
            raise ValueError(
                f"Cannot infer H=W from N={n} (not a perfect square). "
                f"Please pass token_hw=(H,W)."
            )
        h, w = s, s

    return latent.transpose(1, 2).reshape(b, c, h, w)


def _spatial_shuffle(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 3:
        b, n, c = z.shape
        idx = torch.stack([torch.randperm(n, device=z.device) for _ in range(b)], dim=0)
        return z.gather(1, idx.unsqueeze(-1).expand(-1, -1, c))

    if z.dim() == 4:
        b, c, h, w = z.shape
        n = h * w
        tok = z.flatten(2).transpose(1, 2)
        idx = torch.stack([torch.randperm(n, device=z.device) for _ in range(b)], dim=0)
        tok = tok.gather(1, idx.unsqueeze(-1).expand(-1, -1, c))
        return tok.transpose(1, 2).reshape(b, c, h, w)

    raise ValueError(f"Expected z dim 3 or 4, got {tuple(z.shape)}")


class LinearForwardDynamicsProbe(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int = 3,
        predict_residual: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.action_dim = int(action_dim)
        self.predict_residual = bool(predict_residual)

        self.A = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.B = nn.Linear(self.action_dim, self.in_channels, bias=False)

        nn.init.zeros_(self.A.weight)
        if self.A.bias is not None:
            nn.init.zeros_(self.A.bias)
        nn.init.zeros_(self.B.weight)

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor, spatial_shuffle: bool = False) -> torch.Tensor:
        if a_t.dim() != 2:
            raise ValueError(f"Expected a_t as (B,A), got {tuple(a_t.shape)}")

        in_dim = z_t.dim()
        if in_dim == 3:
            b, n, c = z_t.shape
            if c != self.in_channels:
                raise ValueError(f"in_channels mismatch: probe={self.in_channels}, z_t.C={c}")
            if spatial_shuffle:
                z_t = _spatial_shuffle(z_t)
            z_map = _to_bchw(z_t)
            b, c, h, w = z_map.shape
        elif in_dim == 4:
            b, c, h, w = z_t.shape
            if c != self.in_channels:
                raise ValueError(f"in_channels mismatch: probe={self.in_channels}, z_t.C={c}")
            if spatial_shuffle:
                z_t = _spatial_shuffle(z_t)
            z_map = z_t
        else:
            raise ValueError(f"Expected z_t dim 3 or 4, got {tuple(z_t.shape)}")

        a = self.B(a_t).view(b, c, 1, 1)
        dz = self.A(z_map) + a
        out = (z_map + dz) if self.predict_residual else dz

        if in_dim == 3:
            return out.flatten(2).transpose(1, 2)
        return out
