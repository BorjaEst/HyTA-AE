"""Full-Context Masked Training (FCMT) augmentation for spatial maps.

Implements the FCMT protocol where models receive full-context inputs during forward
passes but supervision (loss computation) is restricted to visible regions indicated
by binary masks. This isolates the effects of partial supervision from input deprivation.

Pipeline stages:
    1. Geometry transforms (flips) applied to create spatially augmented targets
    2. Random rectangular masking generates incomplete inputs via element-wise multiplication
    3. Optional cyclic shifts prevent border bias in visibility patterns

Output format:
    - Input sensors: (C+1, H, W) where last channel is the visibility mask
      * Channels 0:C contain masked map values (x * mask)
      * Channel C contains mask (1.0=visible, mask_value=occluded)
    - Target: (C, H, W) clean geometrically-transformed map

Masking semantics (FCMT):
    - Loss computed ONLY on visible positions (mask==1.0)
    - Hidden positions (mask==mask_value) provide context but no supervision
    - Prevents model from learning to overwrite unknown regions
    - Maintains consistent gradient scales across different mask ratios (no normalization)

Key parameters:
    - mask_ratio: Fraction of spatial locations to occlude (e.g., 0.75 = 75% hidden)
    - keep_rects: Number of visible rectangles (union creates irregular patterns)
    - aspect_min/max: Control shape variability of visible regions
    - wrap_shift: Random toroidal shift to uniformize pixel inclusion statistics

Theoretical foundation:
    Adapts masked autoencoding practices (BERT, MAE) to partial supervision setting.
    Unlike MAE (loss on masked regions), FCMT computes loss on visible regions while
    maintaining full-context features for pattern completion.
"""

import math  # added
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
    vflip_p: float = Field(default=0.5, ge=0.0, le=1.0, description="Vertical flip probability")

    # Corruption (incomplete maps)
    mask_ratio: float = Field(default=0.75, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")
    mask_value: float = Field(default=0.0, description="Value used for masked-out locations")

    # Visible-rectangle sampling (prevents always-centered visibility)
    keep_rects: int = Field(default=1, ge=1, le=8, description="Number of visible rectangles to keep (union)")
    aspect_min: float = Field(default=0.5, gt=0.0, description="Min aspect ratio (w/h) for visible rectangles")
    aspect_max: float = Field(default=2.0, gt=0.0, description="Max aspect ratio (w/h) for visible rectangles")
    wrap_shift: bool = Field(default=True, description="Random cyclic shift to mask to uniformize pixel inclusion")


# -------------------------------------------------------------------------------------------
class RandomMask:
    """Randomly keep one or more visible rectangles and mask the rest.

    Mask channel semantics:
      - 1.0 on visible (unmasked) positions
      - mask_value on masked positions

    The masked input is computed as x * mask (values are faded, not replaced).
    """

    def __init__(
        self,
        ratio: float = 0.3,
        value: float = 0.0,
        keep_rects: int = 1,
        aspect_min: float = 0.5,
        aspect_max: float = 2.0,
        wrap_shift: bool = True,
    ):
        self.ratio = float(ratio)
        self.value = float(value)
        self.keep_rects = int(keep_rects)
        self.aspect_min = float(aspect_min)
        self.aspect_max = float(aspect_max)
        self.wrap_shift = bool(wrap_shift)
        # Stores last mask (1,H,W): 1.0 where visible, mask_value where masked
        self.last_mask: Optional[Tensor] = None

    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("RandomMask expects a torch.Tensor")
        if x.dim() != 3:
            raise ValueError(f"Expected (C,H,W) tensor, got shape {tuple(x.shape)}")

        c, h, w = x.shape
        if self.ratio <= 0.0:
            self.last_mask = torch.ones((1, h, w), device=x.device, dtype=x.dtype)
            return x

        visible_ratio = max(0.0, min(1.0, 1.0 - self.ratio))
        keep = torch.zeros((h, w), device=x.device, dtype=x.dtype)

        # Sample one or more visible rectangles with random aspect ratio
        for _ in range(self.keep_rects):
            if visible_ratio <= 0.0:
                continue

            # Log-uniform aspect sampling is common; uniform is fine too
            a = float(torch.empty(1, device=x.device).uniform_(self.aspect_min, self.aspect_max))
            # Ensure h_frac * w_frac ≈ visible_ratio while varying aspect
            h_frac = math.sqrt(visible_ratio / a)
            w_frac = math.sqrt(visible_ratio * a)

            vh = max(1, int(round(h * max(0.0, min(1.0, h_frac)))))
            vw = max(1, int(round(w * max(0.0, min(1.0, w_frac)))))

            max_i = max(0, h - vh)
            max_j = max(0, w - vw)
            i = 0 if max_i == 0 else int(torch.randint(0, max_i + 1, (1,), device=x.device))
            j = 0 if max_j == 0 else int(torch.randint(0, max_j + 1, (1,), device=x.device))

            keep[i : i + vh, j : j + vw] = 1.0

        keep1 = keep.unsqueeze(0)  # (1,H,W)

        # Random cyclic shift to remove border bias
        if self.wrap_shift:
            di = int(torch.randint(0, h, (1,), device=x.device))
            dj = int(torch.randint(0, w, (1,), device=x.device))
            keep1 = torch.roll(keep1, shifts=(di, dj), dims=(-2, -1))

        mask1 = keep1 * 1.0 + (1.0 - keep1) * self.value  # (1,H,W)
        self.last_mask = mask1

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
            keep_rects=self.params.keep_rects,
            aspect_min=self.params.aspect_min,
            aspect_max=self.params.aspect_max,
            wrap_shift=self.params.wrap_shift,
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
    compose_params = ComposeParams(mask_value=0.2)
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
