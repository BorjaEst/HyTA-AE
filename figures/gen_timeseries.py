"""Convert TensorBoard scalar logs to publication-ready vector PDFs.

This module provides utilities to export TensorBoard scalar data into clean,
publication-ready PDF plots with statistical summaries.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensorboard.backend.event_processing import event_accumulator

# -------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------
TABLE_COLUMN_LABELS = ["", "run", "min", "max", "end", "step"]
SWATCH_COLUMN_INDEX = 0
RUN_COLUMN_INDEX = 1
COLOR_SWATCH_SYMBOL = "■"

# Default figure and table dimensions
FIGURE_WIDTH = 7.5
FIGURE_HEIGHT = 5.5
PLOT_TO_TABLE_RATIO = [3, 1]
TABLE_WIDTH = 0.98
SWATCH_MIN_WIDTH = 0.02

# Font sizes
FONT_SIZE_VALUE = 9
FONT_SIZE_TITLE = 11


# -------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------
class Arguments(BaseSettings):
    """Configuration for TensorBoard log export to PDF.

    Attributes
    ----------
    log_dir : str
        Directory containing TensorBoard experiment logs.
    out_dir : str
        Directory where output PDFs will be saved.
    smooth_alpha : float
        EMA smoothing factor in [0, 1). Zero means no smoothing.
    combine_runs : bool
        If True, create one PDF per metric with all runs overlaid.
        If False, create one multi-page PDF per run.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # IO settings
    log_dir: str = Field(default="logs", description="Directory with experiment logs to parse")
    out_dir: str = Field(default="figures/timeseries", description="Directory for output PDFs")

    # Filtering & plotting
    smooth_alpha: float = Field(default=0.0, ge=0.0, lt=1.0, description="EMA smoothing factor (0=no smoothing)")
    combine_runs: bool = Field(default=True, description="Export one PDF per tag across runs")


# -------------------------------------------------------------------------------------------
# TensorBoard Data Loading
# -------------------------------------------------------------------------------------------
def _is_tensorboard_run_dir(path: Path) -> bool:
    """Check whether a directory contains TensorBoard event files.

    Parameters
    ----------
    path: Path
        Directory to inspect.

    Returns
    -------
    bool
        True if the directory has at least one `events.out.tfevents.*` file.
    """

    if not path.is_dir():
        return False
    return any(path.glob("events.out.tfevents.*"))


def _find_run_dirs(root: Path) -> List[Path]:
    """Recursively discover TensorBoard run directories under a root.

    A "run directory" is any directory containing at least one event file.

    Parameters
    ----------
    root: Path
        Root directory to search (e.g., `logs/`).

    Returns
    -------
    List[Path]
        List of discovered run directories, sorted by path.
    """

    run_dirs: List[Path] = []
    if not root.exists():
        return run_dirs

    # Search depth-first; treat any directory with event files as a run dir.
    for path in root.rglob("*"):
        if _is_tensorboard_run_dir(path):
            run_dirs.append(path)

    # Deduplicate in case nested discovery hits same folder via different paths
    run_dirs = sorted(set(run_dirs))
    return run_dirs


def _load_scalars(run_dir: Path) -> Dict[str, Tuple[List[int], List[float]]]:
    """Load scalar series for a given run directory.

    Parameters
    ----------
    run_dir : Path
        Directory containing TensorBoard event files for a single run.

    Returns
    -------
    Dict[str, Tuple[List[int], List[float]]]
        Mapping from tag name to (steps, values).

    Raises
    ------
    RuntimeError
        If tensorboard package is not available.
    """
    if event_accumulator is None:
        raise RuntimeError("tensorboard is not available. Please `pip install tensorboard`.")

    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    series: Dict[str, Tuple[List[int], List[float]]] = {}

    for tag in available_tags:
        scalars = ea.Scalars(tag)
        steps = [int(s.step) for s in scalars]
        values = [float(s.value) for s in scalars]

        # Sort by step to ensure correct ordering
        if steps:
            sorted_data = sorted(zip(steps, values), key=lambda pair: pair[0])
            sorted_steps, sorted_values = zip(*sorted_data)
            series[tag] = (list(sorted_steps), list(sorted_values))

    return series


# -------------------------------------------------------------------------------------------
# Data Processing
# -------------------------------------------------------------------------------------------
def _ema(values: List[float], alpha: float) -> List[float]:
    """Apply exponential moving average smoothing to a series.

    Parameters
    ----------
    values : List[float]
        Input series to smooth.
    alpha : float
        Smoothing factor in [0, 1). Zero returns the original series.

    Returns
    -------
    List[float]
        Smoothed series with same length as input.
    """
    if not values or alpha <= 0.0:
        return list(values)

    smoothed: List[float] = []
    for value in values:
        if not smoothed:
            smoothed.append(value)
        else:
            smoothed.append(alpha * smoothed[-1] + (1.0 - alpha) * value)
    return smoothed


# -------------------------------------------------------------------------------------------
# Matplotlib Configuration
# -------------------------------------------------------------------------------------------# -------------------------------------------------------------------------------------------
# Matplotlib Configuration
# -------------------------------------------------------------------------------------------
def _configure_matplotlib() -> None:
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


def _format_float(value: float) -> str:
    """Format floats compactly for table display.

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
# Table Formatting
# -------------------------------------------------------------------------------------------
def _apply_table_cell_styles(table, num_cols: int, num_data_rows: int) -> None:
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


def _apply_swatch_colors(table, colors: List[str]) -> None:
    """Apply color swatches to the first column of data rows.

    Parameters
    ----------
    table
        Matplotlib table object.
    colors : List[str]
        Color codes for each data row.
    """
    for row_idx, color in enumerate(colors, start=1):
        if (row_idx, SWATCH_COLUMN_INDEX) not in table.get_celld():
            continue
        swatch_cell = table[(row_idx, SWATCH_COLUMN_INDEX)]
        swatch_cell.get_text().set_color(color)
        swatch_cell.get_text().set_ha("left")


def _fit_table_full_width_by_content(
    table,
    ncols: int,
    swatch_col: int = SWATCH_COLUMN_INDEX,
    run_col: int = RUN_COLUMN_INDEX,
    total_width: float = TABLE_WIDTH,
    min_swatch: float = SWATCH_MIN_WIDTH,
) -> None:
    """Fit table columns to content, then scale to occupy nearly full width.

    This preserves relative content-based widths while ensuring the table spans
    almost the entire axes width. The swatch column gets a minimum width, and
    the run column expands to fill remaining space.

    Parameters
    ----------
    table
        Matplotlib table object.
    ncols : int
        Number of columns.
    swatch_col : int
        Index of the color swatch column.
    run_col : int
        Index of the run name column (expands to fill space).
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

    # Validate run column index
    run_col = run_col if 0 <= run_col < ncols else 1

    # Calculate space available for run column
    sum_non_run = sum(widths[col] for col in range(ncols) if col != run_col)
    remaining = max(0.0, total_width) - sum_non_run

    if remaining >= widths[run_col]:
        # Enough space: allocate all remaining width to run column
        new_widths = widths.copy()
        new_widths[run_col] = remaining
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
# Statistics Calculation
# -------------------------------------------------------------------------------------------
def _compute_series_stats(steps: List[int], values: List[float]) -> Dict[str, str]:
    """Compute summary statistics for a data series.

    Parameters
    ----------
    steps : List[int]
        Step numbers.
    values : List[float]
        Corresponding values.

    Returns
    -------
    Dict[str, str]
        Formatted statistics: min, max, end_value, end_step.
    """
    if not values or not steps:
        return {
            "min": _format_float(float("nan")),
            "max": _format_float(float("nan")),
            "end_value": _format_float(float("nan")),
            "end_step": "",
        }

    return {
        "min": _format_float(min(values)),
        "max": _format_float(max(values)),
        "end_value": _format_float(values[-1]),
        "end_step": str(steps[-1]),
    }


def _build_stats_table_row(run_name: str, stats: Dict[str, str]) -> List[str]:
    """Build a table row from run name and statistics.

    Parameters
    ----------
    run_name : str
        Name of the run.
    stats : Dict[str, str]
        Statistics dictionary from _compute_series_stats.

    Returns
    -------
    List[str]
        Table row: [swatch, run_name, min, max, end_value, end_step].
    """
    return [
        COLOR_SWATCH_SYMBOL,
        run_name,
        stats["min"],
        stats["max"],
        stats["end_value"],
        stats["end_step"],
    ]


# -------------------------------------------------------------------------------------------
# Figure Creation
# -------------------------------------------------------------------------------------------
def _create_figure_with_table():
    """Create a figure with plot axes and table axes.

    Returns
    -------
    tuple
        (fig, plot_axes, table_axes)
    """
    return plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={"height_ratios": PLOT_TO_TABLE_RATIO},
    )


def _create_stats_table(table_axes, rows: List[List[str]], colors: List[str]) -> None:
    """Create and style a statistics table under a plot.

    Parameters
    ----------
    table_axes
        Matplotlib axes for the table.
    rows : List[List[str]]
        Data rows for the table.
    colors : List[str]
        Color codes for each row's swatch.
    """
    if not rows:
        return

    table = table_axes.table(
        cellText=rows,
        colLabels=TABLE_COLUMN_LABELS,
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_SIZE_VALUE)

    # Apply column width fitting
    _fit_table_full_width_by_content(
        table,
        ncols=len(TABLE_COLUMN_LABELS),
        swatch_col=SWATCH_COLUMN_INDEX,
        run_col=RUN_COLUMN_INDEX,
        total_width=TABLE_WIDTH,
        min_swatch=SWATCH_MIN_WIDTH,
    )

    # Apply cell styles
    _apply_table_cell_styles(table, len(TABLE_COLUMN_LABELS), len(rows))

    # Apply color swatches
    _apply_swatch_colors(table, colors)

    table_axes.axis("off")


# -------------------------------------------------------------------------------------------
# PDF Export Functions
# -------------------------------------------------------------------------------------------
def _plot_run_to_pdf(
    run_name: str,
    scalars: Dict[str, Tuple[List[int], List[float]]],
    pdf_path: Path,
    smooth_alpha: float,
) -> None:
    """Create a multi-page PDF for one run; one page per tag.

    Parameters
    ----------
    run_name : str
        Name of the run.
    scalars : Dict[str, Tuple[List[int], List[float]]]
        Mapping from tag to (steps, values).
    pdf_path : Path
        Output PDF file path.
    smooth_alpha : float
        EMA smoothing factor.
    """
    if not scalars:
        return

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for tag in sorted(scalars.keys()):
            steps, values = scalars[tag]
            smoothed_values = _ema(values, smooth_alpha)

            # Create figure with plot and table
            fig, (plot_ax, table_ax) = _create_figure_with_table()

            # Plot the series
            plot_ax.plot(steps, smoothed_values, label=run_name, lw=1.5)
            plot_ax.set_xlabel("step")
            plot_ax.set_ylabel(tag.split("/")[-1])
            plot_ax.set_title(f"{tag} — {run_name}")

            # Get line color for swatch
            line_color = plot_ax.lines[-1].get_color() if plot_ax.lines else "black"

            # Build statistics table
            stats = _compute_series_stats(steps, smoothed_values)
            row = _build_stats_table_row(run_name, stats)
            _create_stats_table(table_ax, [row], [line_color])

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _plot_by_tag_across_runs(
    runs: Dict[str, Dict[str, Tuple[List[int], List[float]]]],
    out_dir: Path,
    smooth_alpha: float,
) -> None:
    """Export one PDF per tag, overlaying all runs.

    Parameters
    ----------
    runs : Dict[str, Dict[str, Tuple[List[int], List[float]]]]
        Mapping from run name to its scalar data.
    out_dir : Path
        Output directory for PDFs.
    smooth_alpha : float
        EMA smoothing factor.
    """
    # Collect union of all tags across runs
    all_tags: set[str] = set()
    for run_scalars in runs.values():
        all_tags.update(run_scalars.keys())

    for tag in sorted(all_tags):
        # Create figure with plot and table
        fig, (plot_ax, table_ax) = _create_figure_with_table()

        rows: List[List[str]] = []
        colors: List[str] = []

        for run_name in sorted(runs.keys()):
            run_scalars = runs[run_name]
            if tag not in run_scalars:
                continue

            steps, values = run_scalars[tag]
            smoothed_values = _ema(values, smooth_alpha)

            # Plot the series
            plot_ax.plot(steps, smoothed_values, lw=1.5, label=run_name)
            line_color = plot_ax.lines[-1].get_color() if plot_ax.lines else "black"

            # Build statistics row
            stats = _compute_series_stats(steps, smoothed_values)
            row = _build_stats_table_row(run_name, stats)
            rows.append(row)
            colors.append(line_color)

        plot_ax.set_xlabel("step")
        plot_ax.set_ylabel(tag.split("/")[-1])
        plot_ax.set_title(tag)

        # Create statistics table
        _create_stats_table(table_ax, rows, colors)

        fig.tight_layout()

        # Save to PDF
        tag_safe = tag.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{tag_safe}.pdf")
        plt.close(fig)


# -------------------------------------------------------------------------------------------
# Main Export Function
# -------------------------------------------------------------------------------------------
def export_tensorboard_to_pdf(cfg: Arguments) -> None:
    """Export TensorBoard scalar charts under `cfg.log_dir` into PDF(s).

    Behavior is controlled by `cfg.combine_runs`:
    - True: One PDF per metric with all runs overlaid (default).
    - False: One multi-page PDF per run with a page per metric.

    Parameters
    ----------
    cfg : Arguments
        Configuration specifying directories, smoothing, and export mode.

    Notes
    -----
    - Only scalar metrics are exported as vector graphics (PDF).
    - Image summaries are not exported (raster format not vectorizable).
    - Runs are discovered as any directories containing `events.out.tfevents.*`.
    """
    _configure_matplotlib()

    root = Path(cfg.log_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()

    # Validate input directory
    if not root.exists():
        print(f"[logs2pdf] log_dir does not exist: {root}", file=sys.stderr)
        return

    # Discover run directories
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"[logs2pdf] no TensorBoard runs found in: {root}", file=sys.stderr)
        return

    # Load all scalar data
    runs_data: Dict[str, Dict[str, Tuple[List[int], List[float]]]] = {}
    for run_dir in run_dirs:
        run_name = str(run_dir.relative_to(root))
        try:
            scalars = _load_scalars(run_dir)
            if scalars:
                runs_data[run_name] = scalars
        except Exception as exc:  # pragma: no cover
            print(f"[logs2pdf] failed to load {run_dir}: {exc}", file=sys.stderr)
            continue

    if not runs_data:
        print("[logs2pdf] no scalar tags found to export", file=sys.stderr)
        return

    # Export based on configuration
    if cfg.combine_runs:
        _plot_by_tag_across_runs(runs_data, out_dir, float(cfg.smooth_alpha))
        print(f"[logs2pdf] exported PDFs by-tag under: {out_dir / 'by_tag'}")
    else:
        for run_name, series in runs_data.items():
            run_safe = run_name.replace("/", "_")
            pdf_path = out_dir / "by_run" / f"{run_safe}.pdf"
            _plot_run_to_pdf(
                run_name=run_name,
                scalars=series,
                pdf_path=pdf_path,
                smooth_alpha=float(cfg.smooth_alpha),
            )
        print(f"[logs2pdf] exported PDFs by-run under: {out_dir / 'by_run'}")


# -------------------------------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """CLI entrypoint for TensorBoard log export.

    Example
    -------
    Export all runs combined by tag with smoothing::
    
        python -m figures.logs2pdf --log_dir logs --out_dir figures/tb_pdf \\
            --combine_runs true --smooth_alpha 0.9
    """
    cfg = Arguments()
    export_tensorboard_to_pdf(cfg)
