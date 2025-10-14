"""Script to generate obstacle maps from MiniGrid and export as PDF figures.

This script uses pydantic-settings for configuration management and generates
binary map visualizations in PDF format from MiniGrid environments.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.utils.data import DataLoader

from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.binary_map import BinaryMapFigure, BinaryMapParams


# -------------------------------------------------------------------------------------------
class ScriptSettings(BaseSettings):
    """Configuration settings for obstacle map generation script."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Data generation settings
    env_id: str = Field(default="MiniGrid-MultiRoom-N6-v0", description="MiniGrid env ID to use")
    seed: int = Field(default=42, ge=0, description="Random seed for reproducible generation")
    invert_walls: bool = Field(default=False, description="Invert wall representation (1=free, 0=wall)")
    n_samples: int = Field(default=200, ge=1, description="Number of obstacle maps to generate")
    batch_size: int = Field(default=4, ge=1, le=16, description="Batch size for data loading")

    # Visualization settings
    n_display: int = Field(default=4, ge=1, le=16, description="Number of maps to display in grid")
    grid_rows: int = Field(default=2, ge=1, le=4, description="Number of rows in display grid")
    grid_cols: int = Field(default=2, ge=1, le=4, description="Number of columns in display grid")

    # Figure settings
    fig_width: float = Field(default=12.0, gt=0.0, description="Figure width in inches")
    fig_height: float = Field(default=8.0, gt=0.0, description="Figure height in inches")
    fig_dpi: Optional[int] = Field(default=None, ge=72, le=600, description="Figure DPI for output")
    fig_title: str = Field(default="MiniGrid Obstacle Maps - Sample Variations", description="Main title")
    show_grid: bool = Field(default=True, description="Show grid lines on maps")
    show_frame: bool = Field(default=True, description="Show axes frame and ticks")

    # Output settings
    output_path: Path = Field(default=Path("obstacle_maps_output.pdf"), description="Output file path")
    bbox_inches: str = Field(default="tight", description="Bounding box setting for savefig")


# -------------------------------------------------------------------------------------------
def generate_obstacle_maps(settings: ScriptSettings) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate obstacle maps using configured parameters.

    Args:
        settings: Script configuration settings

    Returns:
        Tuple of (inputs, targets) tensors from first batch
    """
    print(f"Generating {settings.n_samples} obstacle maps using {settings.env_id}...")

    # Create data parameters
    data_params = DataParams(
        env_id=settings.env_id,
        seed=settings.seed,
        invert_walls=settings.invert_walls,
    )

    # Create generator and dataset
    generator = DataGenerator(data_params)
    dataset = generator(n_samples=settings.n_samples)

    # Create DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False)

    # Get first batch
    batch = next(iter(dataloader))
    inputs, targets = batch

    # Report statistics
    density = torch.mean(inputs, dim=(1, 2)).cpu().numpy()
    print(f"Generated batch: {inputs.shape}")
    print(f"Obstacle density range: {density.min():.3f} - {density.max():.3f}")

    return inputs, targets


# -------------------------------------------------------------------------------------------
def create_visualization(inputs: torch.Tensor, settings: ScriptSettings) -> plt.Figure:
    """Create multi-panel visualization of obstacle maps.

    Args:
        inputs: Tensor of obstacle maps (N, H, W)
        settings: Script configuration settings

    Returns:
        Matplotlib Figure object
    """
    print("Creating visualization...")

    # Configure binary map visualization
    params_dict = {
        "fig_width": settings.fig_width,
        "fig_height": settings.fig_height,
        "title": "",  # Individual titles set per subplot
        "grid": settings.show_grid,
        "frame": settings.show_frame,
    }

    # Only add fig_dpi if it's not None
    if settings.fig_dpi is not None:
        params_dict["fig_dpi"] = settings.fig_dpi

    figure_params = BinaryMapParams(**params_dict)

    # Create figure instance
    figure = BinaryMapFigure(figure_params)

    # Create subplot figure
    fig, axes = plt.subplots(
        settings.grid_rows,
        settings.grid_cols,
        figsize=(settings.fig_width, settings.fig_height),
    )
    fig.suptitle(settings.fig_title, fontsize=14, fontweight="bold")

    # Flatten axes for easier iteration
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    # Plot maps on subplots
    n_display = min(settings.n_display, len(inputs))
    for i, ax in enumerate(axes_flat):
        if i < n_display:
            # Extract single map (H, W) from batch
            single_map = inputs[i]

            # Use BinaryMapFigure to plot on provided axes
            figure.plot(single_map, ax=ax)
            ax.set_title(f"Sample {i+1}", fontsize=10)
        else:
            ax.axis("off")

    plt.tight_layout()

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
if __name__ == "__main__":
    """Main script execution."""
    # Load settings from environment and defaults
    settings = ScriptSettings()

    print("=== Obstacle Map Generation Script ===")
    print(f"Environment: {settings.env_id}")
    print(f"Seed: {settings.seed}")
    print(f"Samples: {settings.n_samples}")
    print(f"Output: {settings.output_path}")
    print()

    # Generate obstacle maps
    inputs, targets = generate_obstacle_maps(settings)

    # Create visualization
    fig = create_visualization(inputs, settings)

    # Save to PDF
    save_figure(fig, settings)

    print()
    print("=== Script Configuration ===")
    print("Settings can be overridden with environment variables:")
    print("  OBSTACLE_MAPS_ENV_ID='MiniGrid-Empty-8x8-v0'")
    print("  OBSTACLE_MAPS_SEED=123")
    print("  OBSTACLE_MAPS_N_SAMPLES=100")
    print("  OBSTACLE_MAPS_OUTPUT_PATH='my_maps.pdf'")
    print()
    print("✓ Script completed successfully!")
