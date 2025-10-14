"""Script to generate masked obstacle maps and export as PDF figures.

This script generates incomplete (masked) obstacle maps using MiniGrid environments
with augmentation, visualizing the original map, mask pattern, and masked input.
Uses pydantic-settings for CLI argument parsing.
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import DataLoader

from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.masked_map import MaskedMapFigure, MaskedMapParams


# -------------------------------------------------------------------------------------------
class ScriptSettings(BaseSettings):
    """Configuration settings for masked obstacle map generation script."""

    model_config = SettingsConfigDict(cli_parse_args=True, extra="forbid")

    # Data generation settings
    env_id: str = Field(default="MiniGrid-MultiRoom-N6-v0", description="MiniGrid environment ID")
    seed: int = Field(default=16, ge=0, description="Random seed for reproducible generation")
    invert_walls: bool = Field(default=False, description="Invert wall representation (1=free, 0=wall)")
    n_samples: int = Field(default=200, ge=1, description="Number of obstacle maps to generate")
    batch_size: int = Field(default=4, ge=1, le=16, description="Batch size for data loading")

    # Augmentation settings
    mask_ratio: float = Field(default=0.65, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")
    mask_value: float = Field(default=0.0, description="Value used for masked-out locations")
    keep_rects: int = Field(default=1, ge=1, le=8, description="Number of visible rectangles to keep")
    aspect_min: float = Field(default=0.5, gt=0.0, description="Min aspect ratio (w/h) for visible rectangles")
    aspect_max: float = Field(default=2.0, gt=0.0, description="Max aspect ratio (w/h) for visible rectangles")
    hflip_p: float = Field(default=0.5, ge=0.0, le=1.0, description="Horizontal flip probability")
    vflip_p: float = Field(default=0.5, ge=0.0, le=1.0, description="Vertical flip probability")
    wrap_shift: bool = Field(default=True, description="Random cyclic shift of mask")

    # Visualization settings
    n_display: int = Field(default=4, ge=1, le=16, description="Number of maps to display")
    samples_per_row: int = Field(default=2, ge=1, le=8, description="Number of samples per row")
    map_size: float = Field(default=2.0, ge=1.0, le=8.0, description="Size of each map in inches")
    cmap_data: str = Field(default="Purples", description="Colormap for data maps")
    cmap_mask: str = Field(default="Blues", description="Colormap for mask pattern")
    add_colorbar: bool = Field(default=False, description="Add colorbar to plots")
    show_axes: bool = Field(default=False, description="Show axis ticks and labels")

    # Figure settings
    fig_width: float = Field(default=6.8, gt=0.0, description="Minimum figure width in inches")
    fig_height: float = Field(default=5.2, gt=0.0, description="Minimum figure height in inches")
    fig_dpi: Optional[int] = Field(default=None, ge=72, le=600, description="Figure DPI (None for vector)")
    fig_title: str = Field(default="Masked Obstacle Maps", description="Main figure title")

    # Output settings
    output_path: Path = Field(default=Path("masked_maps_output.pdf"), description="Output PDF file path")
    bbox_inches: str = Field(default="tight", description="Bounding box setting for savefig")


# -------------------------------------------------------------------------------------------
def generate_masked_maps(settings: ScriptSettings) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate masked obstacle maps using configured parameters.

    Args:
        settings: Script configuration settings

    Returns:
        Tuple of (sensors, targets) tensors from first batch
        sensors: (N, C+1, H, W) with last channel as mask
        targets: (N, C, H, W) clean targets
    """
    print(f"Generating {settings.n_samples} masked obstacle maps using {settings.env_id}...")
    print(f"Mask ratio: {settings.mask_ratio:.1%}, Keep rects: {settings.keep_rects}")

    # Create augmentation pipeline
    compose_params = ComposeParams(
        hflip_p=settings.hflip_p,
        vflip_p=settings.vflip_p,
        mask_ratio=settings.mask_ratio,
        mask_value=settings.mask_value,
        keep_rects=settings.keep_rects,
        aspect_min=settings.aspect_min,
        aspect_max=settings.aspect_max,
        wrap_shift=settings.wrap_shift,
    )
    augmentation = Augmentation(compose_params)

    # Create data parameters
    data_params = DataParams(
        env_id=settings.env_id,
        seed=settings.seed,
        invert_walls=settings.invert_walls,
    )

    # Create generator and dataset with augmentation
    generator = DataGenerator(data_params, transform=augmentation)
    dataset = generator(n_samples=settings.n_samples)

    # Create DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    # Get first batch
    batch = next(iter(dataloader))
    sensors, targets = batch

    # Report statistics
    visible_ratio = torch.mean((sensors[:, -1] > 0.5).float(), dim=(1, 2)).cpu().numpy()
    print(f"Generated batch - Sensors: {sensors.shape}, Targets: {targets.shape}")
    print(f"Visible ratio range: {visible_ratio.min():.1%} - {visible_ratio.max():.1%}")

    return sensors, targets


# -------------------------------------------------------------------------------------------
def create_visualization(
    sensors: torch.Tensor,
    targets: torch.Tensor,
    settings: ScriptSettings,
) -> plt.Figure:
    """Create 3-panel visualization of masked obstacle maps.

    Args:
        sensors: Sensor data (N, C+1, H, W) with last channel as mask
        targets: Target data (N, C, H, W)
        settings: Script configuration settings

    Returns:
        Matplotlib Figure object
    """
    print("Creating visualization...")

    # Configure masked map visualization
    params_dict = {
        "title": settings.fig_title,
        "n_samples": settings.n_display,
        "samples_per_row": settings.samples_per_row,
        "map_size": settings.map_size,
        "fig_width": settings.fig_width,
        "fig_height": settings.fig_height,
        "cmap_data": settings.cmap_data,
        "cmap_mask": settings.cmap_mask,
        "add_colorbar": settings.add_colorbar,
        "show_axes": settings.show_axes,
    }

    # Only add fig_dpi if it's not None
    if settings.fig_dpi is not None:
        params_dict["fig_dpi"] = settings.fig_dpi

    figure_params = MaskedMapParams(**params_dict)

    # Create figure instance
    figure = MaskedMapFigure(figure_params)

    # Plot the masked maps
    fig = figure.plot(sensors, targets)

    return fig


# -------------------------------------------------------------------------------------------
def save_figure(fig: plt.Figure, settings: ScriptSettings) -> None:
    """Save figure to PDF file.

    Args:
        fig: Matplotlib Figure to save
        settings: Script configuration settings
    """
    output_path = settings.output_path

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PDF
    fig.savefig(
        output_path,
        format="pdf",
        dpi=settings.fig_dpi,
        bbox_inches=settings.bbox_inches,
    )

    print(f"✓ Saved visualization to {output_path.absolute()}")


# -------------------------------------------------------------------------------------------
def print_usage() -> None:
    """Print usage information and examples."""
    print("\n=== Usage Examples ===")
    print("\n1. Generate with default settings (2 samples, compact layout):")
    print("   python masked_maps.py")
    print("\n2. Customize mask ratio and visible rectangles:")
    print("   python masked_maps.py --mask_ratio 0.8 --keep_rects 2")
    print("\n3. Display 4 samples with smaller maps:")
    print("   python masked_maps.py --samples_per_row 4 --batch_size 4 --map_size 2.0")
    print("\n4. Change colormaps:")
    print("   python masked_maps.py --cmap_data Greens --cmap_mask Reds")
    print("\n5. Larger maps with colorbar:")
    print("   python masked_maps.py --map_size 4.0 --add_colorbar true")
    print("\n6. Show help:")
    print("   python masked_maps.py --help")


# -------------------------------------------------------------------------------------------
def main() -> None:
    """Main script execution."""
    # Parse CLI arguments and load settings
    try:
        settings = ScriptSettings()
    except Exception as e:
        print(f"Error parsing arguments: {e}", file=sys.stderr)
        print_usage()
        sys.exit(1)

    print("=== Masked Obstacle Map Generation Script ===")
    print(f"Environment: {settings.env_id}")
    print(f"Seed: {settings.seed}")
    print(f"Samples: {settings.n_samples}")
    print(f"Mask Ratio: {settings.mask_ratio:.1%}")
    print(f"Output: {settings.output_path}")
    print()

    # Generate masked obstacle maps
    sensors, targets = generate_masked_maps(settings)

    # Create visualization
    fig = create_visualization(sensors, targets, settings)

    # Save to PDF
    save_figure(fig, settings)

    print()
    print("✓ Script completed successfully!")
    print_usage()


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
