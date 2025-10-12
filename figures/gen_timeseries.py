"""Convert TensorBoard scalar logs to publication-ready vector PDFs.

This module provides utilities to export TensorBoard scalar data into clean,
publication-ready PDF plots with statistical summaries.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensorboard.backend.event_processing import event_accumulator

from ehc_sn.utils.figures import (
    COLOR_SWATCH_SYMBOL,
    SWATCH_COLUMN_INDEX,
    SWATCH_MIN_WIDTH,
    TABLE_WIDTH,
    apply_swatch_colors,
    apply_table_cell_styles,
    assign_folder_based_colors,
    configure_matplotlib,
    create_figure_with_table,
    fit_table_to_full_width,
    format_float,
)

# -------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------
TABLE_COLUMN_LABELS = ["", "run", "min", "max", "end", "step"]
RUN_COLUMN_INDEX = 1


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
    log_scale: bool = Field(default=False, description="Use logarithmic scale for y-axis")


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
            "min": format_float(float("nan")),
            "max": format_float(float("nan")),
            "end_value": format_float(float("nan")),
            "end_step": "",
        }

    return {
        "min": format_float(min(values)),
        "max": format_float(max(values)),
        "end_value": format_float(values[-1]),
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
    table.set_fontsize(9)

    # Apply column width fitting
    fit_table_to_full_width(
        table,
        ncols=len(TABLE_COLUMN_LABELS),
        swatch_col=SWATCH_COLUMN_INDEX,
        expandable_col=RUN_COLUMN_INDEX,
        total_width=TABLE_WIDTH,
        min_swatch=SWATCH_MIN_WIDTH,
    )

    # Apply cell styles
    apply_table_cell_styles(table, len(TABLE_COLUMN_LABELS), len(rows))

    # Apply color swatches
    apply_swatch_colors(table, colors)

    table_axes.axis("off")


# -------------------------------------------------------------------------------------------
# PDF Export Functions
# -------------------------------------------------------------------------------------------
def _plot_run_to_pdf(
    run_name: str,
    scalars: Dict[str, Tuple[List[int], List[float]]],
    pdf_path: Path,
    smooth_alpha: float,
    log_scale: bool = False,
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
    log_scale : bool
        If True, use logarithmic scale for y-axis.
    """
    if not scalars:
        return

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for tag in sorted(scalars.keys()):
            steps, values = scalars[tag]
            smoothed_values = _ema(values, smooth_alpha)

            # Create figure with plot and table
            fig, (plot_ax, table_ax) = create_figure_with_table(num_rows=1)

            # Plot the series
            plot_ax.plot(steps, smoothed_values, label=run_name, lw=1.5)
            plot_ax.set_xlabel("step")
            plot_ax.set_ylabel(tag.split("/")[-1])
            plot_ax.set_title(f"{tag} — {run_name}")
            if log_scale:
                plot_ax.set_yscale("log")

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
    log_scale: bool = False,
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
    log_scale : bool
        If True, use logarithmic scale for y-axis.
    """
    # Collect union of all tags across runs
    all_tags: set[str] = set()
    for run_scalars in runs.values():
        all_tags.update(run_scalars.keys())

    # Precompute colors per run for consistent coloring across all tags
    run_color = assign_folder_based_colors(sorted(runs.keys()))

    for tag in sorted(all_tags):
        # Build entries first to know table size
        entries: List[tuple[str, List[int], List[float], str]] = []  # (run_name, steps, smoothed, color)
        for run_name in sorted(runs.keys()):
            run_scalars = runs[run_name]
            if tag not in run_scalars:
                continue
            steps, values = run_scalars[tag]
            smoothed_values = _ema(values, smooth_alpha)
            c = run_color.get(run_name, None)
            entries.append((run_name, steps, smoothed_values, c if c is not None else "black"))

        # Create figure sized to number of table rows
        fig, (plot_ax, table_ax) = create_figure_with_table(num_rows=len(entries))

        rows: List[List[str]] = []
        colors: List[str] = []

        for run_name, steps, smoothed_values, line_color in entries:
            # Plot the series
            plot_ax.plot(steps, smoothed_values, lw=1.5, label=run_name, color=line_color)

            # Build statistics row
            stats = _compute_series_stats(steps, smoothed_values)
            row = _build_stats_table_row(run_name, stats)
            rows.append(row)
            colors.append(line_color)

        plot_ax.set_xlabel("step")
        plot_ax.set_ylabel(tag.split("/")[-1])
        plot_ax.set_title(tag)
        if log_scale:
            plot_ax.set_yscale("log")

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
    configure_matplotlib()

    root = Path(cfg.log_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()

    # Validate input directory
    if not root.exists():
        print(f"[gen_timeseries] log_dir does not exist: {root}", file=sys.stderr)
        return

    # Discover run directories
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"[gen_timeseries] no TensorBoard runs found in: {root}", file=sys.stderr)
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
            print(f"[gen_timeseries] failed to load {run_dir}: {exc}", file=sys.stderr)
            continue

    if not runs_data:
        print("[gen_timeseries] no scalar tags found to export", file=sys.stderr)
        return

    # Export based on configuration
    if cfg.combine_runs:
        by_tag_dir = out_dir / "by_tag"
        _plot_by_tag_across_runs(runs_data, by_tag_dir, float(cfg.smooth_alpha), cfg.log_scale)
        print(f"[gen_timeseries] exported PDFs by-tag under: {by_tag_dir}")
    else:
        for run_name, series in runs_data.items():
            run_safe = run_name.replace("/", "_")
            pdf_path = out_dir / "by_run" / f"{run_safe}.pdf"
            _plot_run_to_pdf(
                run_name=run_name,
                scalars=series,
                pdf_path=pdf_path,
                smooth_alpha=float(cfg.smooth_alpha),
                log_scale=cfg.log_scale,
            )
        print(f"[gen_timeseries] exported PDFs by-run under: {out_dir / 'by_run'}")


# -------------------------------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """CLI entrypoint for TensorBoard log export.

    Example
    -------
    Export all runs combined by tag with smoothing::

        python -m figures.gen_timeseries --log_dir logs --out_dir figures/timeseries \\
            --combine_runs true --smooth_alpha 0.9
    """
    cfg = Arguments()
    export_tensorboard_to_pdf(cfg)
