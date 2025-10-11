"""Shared utilities for figure generation modules.

This module contains common functionality used by both gen_timeseries and gen_latenteval
to avoid code duplication and improve maintainability.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# -------------------------------------------------------------------------------------------
# Shared Constants
# -------------------------------------------------------------------------------------------
SWATCH_COLUMN_INDEX = 0
COLOR_SWATCH_SYMBOL = "■"

# Default figure and table dimensions
FIGURE_WIDTH = 7.5
TABLE_WIDTH = 0.98
SWATCH_MIN_WIDTH = 0.02

# Font sizes
FONT_SIZE_VALUE = 9
FONT_SIZE_TITLE = 11

# Dynamic sizing for plots with many legend/table rows
BASE_PLOT_HEIGHT_IN = 3.6
TABLE_ROW_HEIGHT_IN = 0.25
TABLE_HEADER_HEIGHT_IN = 0.35


# -------------------------------------------------------------------------------------------
# Matplotlib Configuration
# -------------------------------------------------------------------------------------------
def configure_matplotlib() -> None:
    """Set Matplotlib defaults for publication-ready vector output."""
    mpl.rcParams["pdf.fonttype"] = 42  # Embed TrueType fonts
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.25

    # Consistent font sizes
    mpl.rcParams["font.size"] = FONT_SIZE_VALUE
    mpl.rcParams["axes.labelsize"] = FONT_SIZE_VALUE
    mpl.rcParams["axes.titlesize"] = FONT_SIZE_TITLE
    mpl.rcParams["xtick.labelsize"] = FONT_SIZE_VALUE
    mpl.rcParams["ytick.labelsize"] = FONT_SIZE_VALUE
    mpl.rcParams["legend.fontsize"] = FONT_SIZE_VALUE


def format_float(value: float) -> str:
    """Format floats compactly for display.

    Parameters
    ----------
    value : float
        Number to format.

    Returns
    -------
    str
        Formatted string with up to 4 significant digits.
    """
    try:
        return f"{value:.4g}"
    except Exception:
        return str(value)


# -------------------------------------------------------------------------------------------
# Path Utilities
# -------------------------------------------------------------------------------------------
def extract_top_level_folder(run_name: str) -> str:
    """Extract the top-level folder from a run name like 'folder/sub/run'.

    Parameters
    ----------
    run_name : str
        Relative run path (root-relative), with '/' separators.

    Returns
    -------
    str
        The first path component, or the entire string if no '/' present.
    """
    if not run_name:
        return ""
    parts = run_name.split("/")
    return parts[0] if parts else run_name


# -------------------------------------------------------------------------------------------
# Color Mapping Utilities
# -------------------------------------------------------------------------------------------
def assign_folder_based_colors(
    run_or_group_names: List[str],
    extract_folder_fn=None,
) -> Dict[str, str]:
    """Assign deterministic colors with grouping by top-level folder.

    Different folders get distinct base hues (high contrast). Items within the
    same folder get shade variations of that hue (lower contrast).

    Parameters
    ----------
    run_or_group_names : List[str]
        Item identifiers (e.g., run names or group keys).
    extract_folder_fn : callable, optional
        Function to extract folder from item name. Defaults to extract_top_level_folder.

    Returns
    -------
    Dict[str, str]
        Mapping item_name -> hex color string.
    """
    if not run_or_group_names:
        return {}

    if extract_folder_fn is None:
        extract_folder_fn = extract_top_level_folder

    # Group items by their top-level folder
    folder_groups: Dict[str, List[str]] = {}
    for name in run_or_group_names:
        folder = extract_folder_fn(name)
        folder_groups.setdefault(folder, []).append(name)

    folder_names = sorted(folder_groups.keys())
    n_folders = max(1, len(folder_names))

    # Evenly spaced base hues for high contrast across folders
    hue_offset = 0.07
    base_hues = {f: (hue_offset + i / n_folders) % 1.0 for i, f in enumerate(folder_names)}

    # Four distinct variants within each folder: vary saturation and brightness
    variants: List[tuple[float, float]] = [
        (0.95, 0.90),  # vivid and bright
        (0.60, 0.90),  # pastel bright
        (0.85, 0.70),  # vivid mid-bright
        (1.00, 0.55),  # dark vivid
    ]
    # Order from bright to dark for better visual progression
    variants = sorted(variants, key=lambda sv: (sv[1], sv[0]), reverse=True)

    color_map: Dict[str, str] = {}
    for folder in folder_names:
        items_in_folder = sorted(folder_groups[folder])
        hue = base_hues[folder]
        for idx, name in enumerate(items_in_folder):
            sat, val = variants[idx % len(variants)]
            rgb = mcolors.hsv_to_rgb((hue, sat, val))
            color_map[name] = mcolors.to_hex(rgb)

    return color_map


# -------------------------------------------------------------------------------------------
# Table Styling Utilities
# -------------------------------------------------------------------------------------------
def apply_table_cell_styles(table, num_cols: int, num_data_rows: int) -> None:
    """Apply consistent styling to all table cells.

    Parameters
    ----------
    table
        Matplotlib table object.
    num_cols : int
        Number of columns in the table.
    num_data_rows : int
        Number of data rows (excluding header).
    """
    # Style header row (row 0)
    for col in range(num_cols):
        if (0, col) not in table.get_celld():
            continue
        header_cell = table[(0, col)]
        header_cell.get_text().set_weight("bold")
        header_cell.get_text().set_ha("left")
        header_cell.set_edgecolor("none")
        header_cell.set_facecolor("none")

    # Style data rows (rows 1+)
    for row in range(1, num_data_rows + 1):
        for col in range(num_cols):
            if (row, col) not in table.get_celld():
                continue
            data_cell = table[(row, col)]
            data_cell.get_text().set_ha("left")
            data_cell.set_edgecolor("none")
            data_cell.set_facecolor("none")


def apply_swatch_colors(table, colors: List[str], swatch_col: int = SWATCH_COLUMN_INDEX) -> None:
    """Apply color swatches to the specified column of data rows.

    Parameters
    ----------
    table
        Matplotlib table object.
    colors : List[str]
        Color codes for each data row.
    swatch_col : int
        Index of the swatch column.
    """
    for row_idx, color in enumerate(colors, start=1):
        if (row_idx, swatch_col) not in table.get_celld():
            continue
        swatch_cell = table[(row_idx, swatch_col)]
        swatch_cell.get_text().set_color(color)
        swatch_cell.get_text().set_ha("left")


def fit_table_to_full_width(
    table,
    ncols: int,
    swatch_col: int = SWATCH_COLUMN_INDEX,
    expandable_col: int = 1,
    total_width: float = TABLE_WIDTH,
    min_swatch: float = SWATCH_MIN_WIDTH,
) -> None:
    """Fit table columns to content, then scale to occupy nearly full width.

    This preserves relative content-based widths while ensuring the table spans
    almost the entire axes width. The swatch column gets a minimum width, and
    the expandable column fills remaining space.

    Parameters
    ----------
    table
        Matplotlib table object.
    ncols : int
        Number of columns.
    swatch_col : int
        Index of the color swatch column.
    expandable_col : int
        Index of the column that expands to fill space (e.g., "run" or "Group").
    total_width : float
        Target total width as fraction of axes width.
    min_swatch : float
        Minimum width for swatch column as fraction of axes width.
    """
    # Compute content-based widths
    try:
        table.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass

    # Extract current widths from header row
    widths = [
        table.get_celld().get((0, col), None).get_width() if (0, col) in table.get_celld() else 0.0
        for col in range(ncols)
    ]

    # Enforce minimum swatch width
    if 0 <= swatch_col < ncols:
        widths[swatch_col] = max(widths[swatch_col], min_swatch)

    # Validate expandable column index
    expandable_col = expandable_col if 0 <= expandable_col < ncols else 1

    # Calculate space available for expandable column
    sum_non_expandable = sum(widths[col] for col in range(ncols) if col != expandable_col)
    remaining = max(0.0, total_width) - sum_non_expandable

    if remaining >= widths[expandable_col]:
        # Enough space: allocate all remaining width to expandable column
        new_widths = widths.copy()
        new_widths[expandable_col] = remaining
    else:
        # Not enough space: scale all columns proportionally
        total_current = sum(widths) or 1.0
        scale = total_width / total_current
        new_widths = [w * scale for w in widths]

    # Apply new widths to all cells
    for col in range(ncols):
        for (row, c), cell in table.get_celld().items():
            if c == col:
                cell.set_width(new_widths[col])


# -------------------------------------------------------------------------------------------
# Figure Creation Utilities
# -------------------------------------------------------------------------------------------
def create_figure_with_table(num_rows: int) -> tuple:
    """Create a figure with plot axes and table axes sized for table rows.

    Parameters
    ----------
    num_rows : int
        Number of data rows in the table (excluding header).

    Returns
    -------
    tuple
        (fig, plot_axes, table_axes)
    """
    num_rows = max(0, int(num_rows))
    table_height = TABLE_HEADER_HEIGHT_IN + num_rows * TABLE_ROW_HEIGHT_IN
    plot_height = BASE_PLOT_HEIGHT_IN
    total_height = plot_height + table_height

    return plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH, total_height),
        gridspec_kw={"height_ratios": [plot_height, table_height]},
    )
