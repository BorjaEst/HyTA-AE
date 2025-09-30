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

    Mask channel semantics:
      - 1.0 on visible (unmasked) positions
      - mask_value on masked positions

    The masked input is computed as x * mask (i.e., values are faded, not replaced).
    """

    def __init__(self, ratio: float = 0.3, value: float = 0.0):
        self.ratio = float(ratio)
        self.value = float(value)
        # Stores last mask (1,H,W): 1.0 where visible, mask_value where masked
        self.last_mask: Optional[Tensor] = None

    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("RandomMask expects a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(x.shape)}")

        c, h, w = x.shape
        if self.ratio <= 0.0:
            # No masking: mask is all ones, input unchanged
            self.last_mask = torch.ones((1, h, w), device=x.device, dtype=x.dtype)
            return x

        # Compute visible area ratio; keep a single rectangle with this area
        visible_ratio = max(0.0, min(1.0, 1.0 - self.ratio))

        if visible_ratio <= 0.0:
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

        keep1 = keep.unsqueeze(0)  # (1,H,W)

        # Build mask channel: 1.0 where visible, mask_value where masked
        mask1 = keep1 * 1.0 + (1.0 - keep1) * self.value  # (1,H,W)
        self.last_mask = mask1

        # Apply faded masking to input: x_incomplete = x * mask
        x_incomplete = x * mask1.expand(c, -1, -1)

        return x_incomplete


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
        Returns sensors with shape (C+1,H,W):
          - first C channels: masked input
          - last channel: mask (1.0 visible, mask_value masked)
        """
        y = self._geom(x)  # clean target after geometry
        x_incomplete = self._mask(y)
        mask = self._mask.last_mask
        if mask is None:
            # Fallback: all visible
            mask = torch.ones_like(y[:1])

        sensors = torch.cat([x_incomplete, mask.to(torch.float32)], dim=0)
        return sensors, y


if __name__ == "__main__":
    # Simple test of augmentation pipeline
    print(f"\n--- Testing Augmentation Pipeline ---")
    import matplotlib.pyplot as plt

    from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
    from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure

    # Create data generator with augmentation
    compose_params = ComposeParams(hflip_p=0.0, vflip_p=0.0, mask_ratio=0.6, mask_value=0.2)
    augmentation = Augmentation(compose_params)
    data_params = DataParams(env_id="MiniGrid-MultiRoom-N6-v0", seed=42, invert_walls=False)
    generator = DataGenerator(data_params, transform=augmentation)

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
