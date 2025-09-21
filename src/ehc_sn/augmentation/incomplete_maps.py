"""Augmentation utilities using torchvision v2 for (incomplete) binary maps.

- Geometry ops (flip/resize) are applied first.
- Target is taken as the geometrically transformed map (clean).
- Corruption (masking) is applied only to the input to create incomplete maps.
"""

from typing import Callable, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T


# -------------------------------------------------------------------------------------------
class ComposeParams(BaseModel):
    """Parameters for augmentation composition using torchvision v2."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Geometry
    hflip_p: float = Field(default=0.5, ge=0.0, le=1.0, description="Horizontal flip probability")
    vflip_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Vertical flip probability")
    resize_to: Optional[Tuple[int, int]] = Field(
        default=None, description="Optional (H,W). Use NEAREST to preserve binary semantics"
    )

    # Corruption (incomplete maps)
    mask_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")
    mask_value: float = Field(default=0.0, description="Value used for masked-out locations")
    preserve_walls: bool = Field(
        default=True,
        description="If True, do not mask wall cells (assumes channel 0 == walls)",
    )


# -------------------------------------------------------------------------------------------
class RandomMask:
    """Randomly mask spatial locations to create incomplete maps.

    Assumes input is a float tensor of shape (C, H, W) with binary channels
    (channel 0 == walls). Applies the same mask across channels and optionally
    preserves wall cells.

    This is a simple callable that integrates in v2.Compose.
    """

    def __init__(self, ratio: float = 0.3, value: float = 0.0, preserve_walls: bool = True):
        self.ratio = float(ratio)
        self.value = float(value)
        self.preserve_walls = bool(preserve_walls)

    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("RandomMask expects a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(x.shape)}")

        c, h, w = x.shape
        if self.ratio <= 0.0:
            return x

        # Bernoulli keep mask (1=keep, 0=mask)
        keep = torch.rand((h, w), device=x.device, dtype=x.dtype)
        keep = (keep > self.ratio).to(x.dtype)  # shape (H, W)

        if self.preserve_walls and c >= 1:
            walls = (x[0] > 0.5).to(x.dtype)  # channel 0 assumed walls (binary)
            keep = torch.clamp(keep + walls, max=1.0)

        # Broadcast to channels
        keep_ch = keep.unsqueeze(0).expand(c, -1, -1)

        # Apply mask
        return x * keep_ch + (1.0 - keep_ch) * self.value


# -------------------------------------------------------------------------------------------
class Augmentation:
    """Build and apply a v2 augmentation pipeline for binary maps.

    Pipeline:
      - ToDtype(float32)
      - Geometry ops (flip/resize; NEAREST only)
      - Return target as geometrically transformed map (clean)
      - Apply RandomMask only to input to produce incomplete maps
    """

    def __init__(self, params: ComposeParams):
        self.params = params
        self._geom = self._build_geometry(params)
        self._mask = RandomMask(ratio=params.mask_ratio, value=params.mask_value, preserve_walls=params.preserve_walls)

    # -----------------------------------------------------------------------------------
    def _build_geometry(self, p: ComposeParams) -> T.Compose:
        ops = [T.ToDtype(torch.float32, scale=False)]
        if p.hflip_p > 0:
            ops.append(T.RandomHorizontalFlip(p=p.hflip_p))
        if p.vflip_p > 0:
            ops.append(T.RandomVerticalFlip(p=p.vflip_p))
        if p.resize_to is not None:
            ops.append(T.Resize(p.resize_to, interpolation=InterpolationMode.NEAREST, antialias=False))
        return T.Compose(ops)

    # -----------------------------------------------------------------------------------
    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input map tensor (C,H,W), binary channels with channel 0 as walls.

        Returns:
            (incomplete_input, clean_target)
        """
        # Geometry (random) once
        y = self._geom(x)  # clean target after geometry
        # Corruption only for the input
        x_incomplete = self._mask(y)
        return x_incomplete, y
