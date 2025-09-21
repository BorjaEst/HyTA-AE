"""Augmentation utilities using torchvision v2 for (incomplete) binary maps.

- Geometry ops (flip) are applied first.
- Target is taken as the geometrically transformed map (clean).
- Corruption (masking) is applied only to the input to create incomplete maps.
"""

from typing import Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torchvision.transforms import v2 as T


# -------------------------------------------------------------------------------------------
class ComposeParams(BaseModel):
    """Parameters for augmentation composition using torchvision v2."""

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Geometry
    hflip_p: float = Field(default=0.5, ge=0.0, le=1.0, description="Horizontal flip probability")
    vflip_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Vertical flip probability")

    # Corruption (incomplete maps)
    mask_ratio: float = Field(default=0.6, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")
    mask_value: float = Field(default=0.0, description="Value used for masked-out locations")


# -------------------------------------------------------------------------------------------
class RandomMask:
    """Randomly keep a single visible rectangle and mask the rest.

    Assumes input is a float tensor of shape (C, H, W) with binary channels
    (channel 0 == walls). Applies the same rectangular mask across channels.

    The rectangle area is approximately (1 - ratio) of the map area.
    """

    def __init__(self, ratio: float = 0.3, value: float = 0.0):
        self.ratio = float(ratio)
        self.value = float(value)

    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("RandomMask expects a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(x.shape)}")

        c, h, w = x.shape
        if self.ratio <= 0.0:
            # No masking requested
            return x

        # Compute visible area ratio; keep a single rectangle with this area
        visible_ratio = max(0.0, min(1.0, 1.0 - self.ratio))

        if visible_ratio <= 0.0:
            # Fully masked
            keep = torch.zeros((h, w), device=x.device, dtype=x.dtype)
        else:
            # Keep rectangle with approximately visible_ratio area.
            # Maintain map aspect by scaling both dims by sqrt(visible_ratio).
            scale = visible_ratio**0.5
            vh = max(1, int(round(h * scale)))
            vw = max(1, int(round(w * scale)))

            max_i = max(0, h - vh)
            max_j = max(0, w - vw)
            i = 0 if max_i == 0 else int(torch.randint(0, max_i + 1, (1,), device=x.device))
            j = 0 if max_j == 0 else int(torch.randint(0, max_j + 1, (1,), device=x.device))

            keep = torch.zeros((h, w), device=x.device, dtype=x.dtype)
            keep[i : i + vh, j : j + vw] = 1.0

        keep_ch = keep.unsqueeze(0).expand(c, -1, -1)
        return x * keep_ch + (1.0 - keep_ch) * self.value


# -------------------------------------------------------------------------------------------
class Augmentation:
    """Build and apply a v2 augmentation pipeline for binary maps.

    Pipeline:
      - ToDtype(float32)
      - Geometry ops (flip only)
      - Return target as geometrically transformed map (clean)
      - Apply RandomMask only to input to produce incomplete maps
    """

    def __init__(self, params: Optional[ComposeParams] = None):
        self.params = params or ComposeParams()
        self._geom = self._build_geometry(self.params)
        self._mask = RandomMask(
            ratio=self.params.mask_ratio,
            value=self.params.mask_value,
        )

    # -----------------------------------------------------------------------------------
    def _build_geometry(self, p: ComposeParams) -> T.Compose:
        ops = [T.ToDtype(torch.float32, scale=False)]
        if p.hflip_p > 0:
            ops.append(T.RandomHorizontalFlip(p=p.hflip_p))
        if p.vflip_p > 0:
            ops.append(T.RandomVerticalFlip(p=p.vflip_p))
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


if __name__ == "__main__":
    # Simple test of augmentation pipeline
    print(f"\n--- Testing Augmentation Pipeline ---")
    import matplotlib.pyplot as plt

    from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
    from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure

    # Create data generator with augmentation
    params = DataParams(env_id="MiniGrid-MultiRoom-N6-v0", seed=42, invert_walls=False)
    generator = DataGenerator(params, transform=Augmentation())

    # Generate a batch of samples
    dataset = generator(4)
    print(f"Dataset length: {len(dataset)}")
    inputs, targets = zip(*(dataset[i] for i in range(len(dataset))))
    inputs = torch.stack(inputs)  # (N,C,H,W)
    targets = torch.stack(targets)  # (N,C,H,W)
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

    # Figure reconstruction map comparing inputs and outputs
    fig_reconstruction = ReconstructionMapFigure()
    _ = fig_reconstruction.plot(targets, inputs)
    plt.show()
