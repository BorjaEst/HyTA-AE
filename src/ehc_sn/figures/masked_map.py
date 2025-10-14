"""Masked map visualization for incomplete binary maps with augmentation.

This module provides 3-panel visualization for masked (incomplete) obstacle maps,
showing the original map, mask pattern, and masked input.
Designed for pattern completion research.
"""

from typing import Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import Field
from torch import Tensor

from ehc_sn.core.figure import BaseFigure, FigureParams


# -------------------------------------------------------------------------------------------
class MaskedMapParams(FigureParams):
    """Parameters for masked map visualization."""

    # Layout settings
    samples_per_row: int = Field(default=2, ge=1, le=8, description="Number of samples to display (one sample per row)")
    map_size: float = Field(default=2.0, ge=1.0, le=8.0, description="Size of each map in inches")

    # Figure size constraints (override base FigureParams defaults)
    fig_width: float = Field(default=6.0, ge=3.0, le=24.0, description="Figure width (inches)")
    fig_height: float = Field(default=4.0, ge=2.0, le=24.0, description="Figure height (inches)")

    # Visual settings
    cmap_data: str = Field(default="Purples", description="Colormap for maps")
    cmap_mask: str = Field(default="Blues", description="Colormap for mask")
    interpolation: str = Field(default="nearest", description="Image interpolation")
    add_colorbar: bool = Field(default=False, description="Add colorbar to plots")
    vmin: Optional[float] = Field(default=None, description="Minimum value for colormap")
    vmax: Optional[float] = Field(default=None, description="Maximum value for colormap")
    show_axes: bool = Field(default=False, description="Show axis ticks and labels")


# -------------------------------------------------------------------------------------------
class MaskedMapFigure(BaseFigure):
    """3-panel figure for visualizing masked (incomplete) obstacle maps.

    Each sample is displayed in one row with 3 columns:
    1. Original complete map (target/ground truth)
    2. Mask pattern (1=visible, mask_value=masked)
    3. Masked input (what the network sees)

    The layout is vertical: each sample occupies one full row.

    Designed for pattern completion and spatial memory research.

    Example:
        >>> from ehc_sn.figures import MaskedMapFigure, MaskedMapParams
        >>> from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
        >>> from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
        >>>
        >>> # Generate augmented data
        >>> augmentation = Augmentation(ComposeParams(mask_ratio=0.75))
        >>> generator = DataGenerator(DataParams(), transform=augmentation)
        >>> dataset = generator(n_samples=4)
        >>>
        >>> # Visualize
        >>> params = MaskedMapParams(title="Masked Obstacle Maps")
        >>> figure = MaskedMapFigure(params)
        >>> sensors, targets = dataset[0]  # sensors: (C+1,H,W), targets: (C,H,W)
        >>> fig = figure.plot(sensors, targets)
        >>> fig.show()
    """

    def __init__(self, params: Optional[MaskedMapParams] = None) -> None:
        super().__init__(params or MaskedMapParams())

    @property
    def p(self) -> MaskedMapParams:
        """Access to typed parameters."""
        return self.params  # type: ignore[return-value]

    def plot(self, sensors: Union[Tensor, np.ndarray], targets: Union[Tensor, np.ndarray]) -> Figure:
        """Plot 3-panel masked map visualization with multiple samples per row.

        Args:
            sensors: Sensor data of shape (N, C+1, H, W) where last channel is mask,
                    or (C+1, H, W) for single sample
            targets: Target data of shape (N, C, H, W) or (C, H, W) for single sample

        Returns:
            Matplotlib Figure object

        Raises:
            ValueError: If data shapes are incompatible
        """
        # Convert to numpy and validate
        if isinstance(sensors, Tensor):
            sensors_np = self.to_numpy(sensors)
        else:
            sensors_np = sensors

        if isinstance(targets, Tensor):
            targets_np = self.to_numpy(targets)
        else:
            targets_np = targets

        # Handle single sample case
        if sensors_np.ndim == 3:
            sensors_np = sensors_np[np.newaxis, ...]  # (1, C+1, H, W)
        if targets_np.ndim == 2:
            targets_np = targets_np[np.newaxis, np.newaxis, ...]  # (H, W) → (1, 1, H, W)
        elif targets_np.ndim == 3:
            targets_np = targets_np[:, np.newaxis, ...]  # (N, H, W) → (N, 1, H, W)

        if sensors_np.ndim != 4 or targets_np.ndim != 4:
            raise ValueError(f"Expected 4D tensors, got sensors: {sensors_np.shape}, targets: {targets_np.shape}")

        n_samples = min(sensors_np.shape[0], targets_np.shape[0], self.p.n_samples)

        # Calculate grid layout: one row per sample, 3 columns per sample
        # samples_per_row parameter controls how many samples to show
        n_samples_to_show = min(n_samples, self.p.samples_per_row)
        n_rows = n_samples_to_show  # One row per sample
        n_cols = 3  # 3 panels per sample: original, mask, masked

        # Calculate optimal figure size based on map_size
        # Width: 3 maps per row × map_size, with some padding
        calculated_width = n_cols * self.p.map_size * 1.1  # 10% padding between maps
        calculated_height = n_rows * self.p.map_size * 1.1  # 10% padding between rows

        # Apply constraints from fig_width and fig_height parameters
        fig_width = max(self.p.fig_width, calculated_width) if self.p.fig_width else calculated_width
        fig_height = max(self.p.fig_height, calculated_height) if self.p.fig_height else calculated_height

        # Create figure with calculated size
        from matplotlib import pyplot as plt

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(fig_width, fig_height),
            dpi=self.p.fig_dpi,
            squeeze=False,
            tight_layout=True,
        )
        if self.p.title:
            fig.suptitle(self.p.title)

        for i in range(n_samples_to_show):
            # Each sample gets its own row
            row = i

            # Extract data for this sample
            sensor_sample = sensors_np[i]  # (C+1, H, W)
            target_sample = targets_np[i]  # (C, H, W)

            # Decompose sensors: first C channels are masked input, last is mask
            masked_input = sensor_sample[:-1]  # (C, H, W)
            mask = sensor_sample[-1]  # (H, W)

            # Use first channel for visualization
            original_map = target_sample[0]  # (H, W)
            masked_map = masked_input[0]  # (H, W)

            # Get axes for this sample's 3 panels (row has 3 columns)
            ax_original = axes[row, 0]
            ax_mask = axes[row, 1]
            ax_masked = axes[row, 2]

            # Original map
            im1 = ax_original.imshow(
                original_map,
                cmap=self.p.cmap_data,
                vmin=self.p.vmin,
                vmax=self.p.vmax,
                interpolation=self.p.interpolation,
            )
            if i == 0:  # First sample gets titles
                ax_original.set_title("Original")

            # Mask pattern
            im2 = ax_mask.imshow(
                mask,
                cmap=self.p.cmap_mask,
                vmin=self.p.vmin,
                vmax=self.p.vmax,
                interpolation=self.p.interpolation,
            )
            if i == 0:
                ax_mask.set_title("Mask")

            # Masked input
            im3 = ax_masked.imshow(
                masked_map,
                cmap=self.p.cmap_data,
                vmin=self.p.vmin,
                vmax=self.p.vmax,
                interpolation=self.p.interpolation,
            )
            if i == 0:
                ax_masked.set_title("Masked Input")

            # Configure axes
            for ax in [ax_original, ax_mask, ax_masked]:
                if not self.p.show_axes:
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")

            # Add colorbars if requested
            if self.p.add_colorbar:
                fig.colorbar(im1, ax=ax_original, fraction=0.046, pad=0.04)
                fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.04)
                fig.colorbar(im3, ax=ax_masked, fraction=0.046, pad=0.04)

        return fig


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """Test the masked map figure."""
    import torch

    from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
    from ehc_sn.data.obstacle_maps import DataGenerator, DataParams

    print("=== Testing MaskedMapFigure ===")

    # Create augmented data
    compose_params = ComposeParams(mask_ratio=0.75, mask_value=0.0, keep_rects=2)
    augmentation = Augmentation(compose_params)

    data_params = DataParams(env_id="MiniGrid-MultiRoom-N6-v0", seed=42)
    generator = DataGenerator(data_params, transform=augmentation)
    dataset = generator(n_samples=3)

    # Collect samples
    sensors_list, targets_list = [], []
    for i in range(len(dataset)):
        sensors, targets = dataset[i]
        sensors_list.append(sensors)
        targets_list.append(targets)

    sensors_batch = torch.stack(sensors_list)  # (N, C+1, H, W)
    targets_batch = torch.stack(targets_list)  # (N, C, H, W)

    print(f"Sensors shape: {sensors_batch.shape}")
    print(f"Targets shape: {targets_batch.shape}")

    # Test with default parameters
    params = MaskedMapParams(
        title="Masked Obstacle Maps - Pattern Completion Test",
        n_samples=2,
        samples_per_row=3,
        map_size=2.5,
    )

    figure = MaskedMapFigure(params)
    fig = figure.plot(sensors_batch, targets_batch)

    print("✓ MaskedMapFigure test completed!")

    # Uncomment to display
    import matplotlib.pyplot as plt

    plt.show()
