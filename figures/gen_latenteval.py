"""Compare latent_size hyperparameter across TensorBoard runs.

This module scans TensorBoard log directories, reads hyperparameters from
``hparams.yaml`` files, extracts scalar metrics per run, and generates
publication-ready line plots showing how metrics vary with latent_size.
One PDF is created per metric, with runs grouped by top-level folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensorboard.backend.event_processing import event_accumulator

try:  # YAML is used to read hyperparameters
    import yaml
except Exception:  # pragma: no cover - surfaced at runtime with clear error
    yaml = None  # type: ignore[assignment]


# -------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------
HYPERPARAMETER_KEY = "latent_size"  # The hyperparameter to plot on x-axis

TABLE_COLUMN_LABELS = ["", "group"]
SWATCH_COLUMN_INDEX = 0
GROUP_COLUMN_INDEX = 1
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
# Configuration
# -------------------------------------------------------------------------------------------
class Arguments(BaseSettings):
    """Configuration for latent_size hyperparameter comparison export to PDF.

    Attributes
    ----------
    log_dir : str
        Directory containing TensorBoard experiment logs.
    out_dir : str
        Directory where output PDFs will be saved.
    metric_reduce : str
        How to reduce each metric series into a single value per run: ``last``,
        ``min``, or ``max``.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # IO settings
    log_dir: str = Field(default="logs", description="Directory with experiment logs to parse")
    out_dir: str = Field(default="figures/latenteval", description="Directory for output PDFs")

    # Metric reduction
    metric_reduce: str = Field(default="last", description="Reduction: last|min|max")


# -------------------------------------------------------------------------------------------
# TensorBoard & Filesystem Helpers
# -------------------------------------------------------------------------------------------
def _is_tensorboard_run_dir(path: Path) -> bool:
    """Check whether a directory contains TensorBoard event files."""
    if not path.is_dir():
        return False
    return any(path.glob("events.out.tfevents.*"))


def _find_run_dirs(root: Path) -> List[Path]:
    """Recursively discover TensorBoard run directories under a root."""
    if not root.exists():
        return []
    run_dirs = [p for p in root.rglob("*") if _is_tensorboard_run_dir(p)]
    return sorted(set(run_dirs))


def _find_hparams_file(start_dir: Path) -> Optional[Path]:
    """Find a hparams.yaml file starting from run dir and walking up a few levels."""
    candidates = ("hparams.yaml", "hparams.yml")
    cur = start_dir
    for _ in range(4):
        for name in candidates:
            p = cur / name
            if p.exists():
                return p
        cur = cur.parent
    return None


def _load_hparams(hp_file: Path) -> Dict[str, Any]:
    """Load hyperparameters from a YAML file into a flat dict."""
    if yaml is None:
        raise RuntimeError("PyYAML is not available. Please `pip install pyyaml`.")
    with hp_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Flatten simple two-level dicts (common in Lightning)
    flat: Dict[str, Any] = {}

    def _flatten(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _flatten(key, v)
        else:
            flat[prefix] = obj

    _flatten("", data)
    return flat


def _load_metric_value(run_dir: Path, tag: str, reduce: str) -> Optional[float]:
    """Load a scalar series for ``tag`` and reduce to a single value per run."""
    if event_accumulator is None:
        raise RuntimeError("tensorboard is not available. Please `pip install tensorboard`.")

    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return None

    if tag not in ea.Tags().get("scalars", []):
        return None

    scalars = ea.Scalars(tag)
    if not scalars:
        return None

    values = [float(s.value) for s in scalars]

    if reduce == "min":
        return min(values)
    elif reduce == "max":
        return max(values)
    else:  # "last" default
        return values[-1]


def _load_all_metrics(run_dir: Path) -> List[str]:
    """Load all available scalar metric tags for a run."""
    if event_accumulator is None:
        raise RuntimeError("tensorboard is not available. Please `pip install tensorboard`.")

    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return []

    return ea.Tags().get("scalars", [])


# -------------------------------------------------------------------------------------------
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
    """Format floats compactly for display."""
    try:
        return f"{value:.4g}"
    except Exception:
        return str(value)


# -------------------------------------------------------------------------------------------
# Grouping & Color Mapping
# -------------------------------------------------------------------------------------------
def _top_level_folder(run_name: str) -> str:
    """Extract the top-level folder from a run name like 'folder/sub/run'."""
    if not run_name:
        return ""
    parts = run_name.split("/")
    return parts[0] if parts else run_name


def _build_group_color_map(group_names: List[str]) -> Dict[str, str]:
    """Assign one deterministic color per group using distinct base hues."""
    if not group_names:
        return {}
    group_names = sorted(group_names)
    n_groups = max(1, len(group_names))
    hue_offset = 0.07
    base_hues = [(hue_offset + i / n_groups) % 1.0 for i in range(n_groups)]
    # Pick the vivid bright variant
    sat, val = (0.95, 0.90)
    out: Dict[str, str] = {}
    for name, hue in zip(group_names, base_hues):
        rgb = mcolors.hsv_to_rgb((hue, sat, val))
        out[name] = mcolors.to_hex(rgb)
    return out


# -------------------------------------------------------------------------------------------
# Table Utilities
# -------------------------------------------------------------------------------------------
def _apply_table_cell_styles(table, num_cols: int, num_data_rows: int) -> None:
    """Apply consistent styling to all table cells."""
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
    """Apply color swatches to the first column of data rows."""
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
    group_col: int = GROUP_COLUMN_INDEX,
    total_width: float = TABLE_WIDTH,
    min_swatch: float = SWATCH_MIN_WIDTH,
) -> None:
    """Fit table columns to content, then scale to occupy nearly full width."""
    try:
        table.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass

    widths = [
        table.get_celld().get((0, col), None).get_width() if (0, col) in table.get_celld() else 0.0
        for col in range(ncols)
    ]

    if 0 <= swatch_col < ncols:
        widths[swatch_col] = max(widths[swatch_col], min_swatch)

    group_col = group_col if 0 <= group_col < ncols else 1

    sum_non_group = sum(widths[col] for col in range(ncols) if col != group_col)
    remaining = max(0.0, total_width) - sum_non_group

    if remaining >= widths[group_col]:
        new_widths = widths.copy()
        new_widths[group_col] = remaining
    else:
        total_current = sum(widths) or 1.0
        scale = total_width / total_current
        new_widths = [w * scale for w in widths]

    for col in range(ncols):
        for (row, c), cell in table.get_celld().items():
            if c == col:
                cell.set_width(new_widths[col])


# -------------------------------------------------------------------------------------------
# Figure Creation
# -------------------------------------------------------------------------------------------
def _create_figure_with_table(num_rows: int) -> tuple:
    """Create a figure with plot axes and table axes sized for table rows."""
    num_rows = max(0, int(num_rows))
    table_h = TABLE_HEADER_HEIGHT_IN + num_rows * TABLE_ROW_HEIGHT_IN
    plot_h = BASE_PLOT_HEIGHT_IN
    total_h = plot_h + table_h

    return plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH, total_h),
        gridspec_kw={"height_ratios": [plot_h, table_h]},
    )


def _create_legend_table(table_axes, rows: List[List[str]], colors: List[str]) -> None:
    """Create and style a legend table under a plot."""
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
        group_col=GROUP_COLUMN_INDEX,
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
def _plot_metric_by_latent_size(
    metric_tag: str,
    runs_data: Dict[str, Tuple[Dict[str, Any], float]],
    group_color: Dict[str, str],
    out_path: Path,
) -> None:
    """Create one PDF for a metric showing latent_size (x) vs metric (y) per group.

    Parameters
    ----------
    metric_tag : str
        The metric tag to plot.
    runs_data : Dict[str, Tuple[Dict[str, Any], float]]
        Mapping from run_name to (hparams, metric_value).
    group_color : Dict[str, str]
        Mapping from group name to color.
    out_path : Path
        Output PDF file path.
    """
    # Aggregate per-group: collect (latent_size, metric_value) per run
    group_points: Dict[str, List[Tuple[float, float]]] = {}

    for run_name, (hparams, metric_val) in runs_data.items():
        latent_size = hparams.get(HYPERPARAMETER_KEY, None)
        if latent_size is None:
            continue
        try:
            latent_size_float = float(latent_size)
        except Exception:
            continue

        g = _top_level_folder(run_name)
        group_points.setdefault(g, []).append((latent_size_float, metric_val))

    # Filter out groups with no points
    group_entries = [(g, pts) for g, pts in group_points.items() if pts]
    if not group_entries:
        return

    # Create figure sized to number of groups (legend rows)
    fig, (plot_ax, table_ax) = _create_figure_with_table(num_rows=len(group_entries))

    # Draw one line per group: sort by latent_size
    for g, pts in sorted(group_entries, key=lambda it: it[0]):
        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        c = group_color.get(g, "black")
        plot_ax.plot(
            xs,
            ys,
            lw=1.8,
            color=c,
            label=g,
            marker="o",
            markersize=4,
            markerfacecolor=c,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

    plot_ax.set_xlabel(HYPERPARAMETER_KEY)
    plot_ax.set_ylabel(metric_tag.split("/")[-1])
    plot_ax.set_title(metric_tag)

    # Build legend table rows (swatch, group)
    rows: List[List[str]] = []
    row_colors: List[str] = []
    for g, _pts in sorted(group_entries, key=lambda it: it[0]):
        rows.append([COLOR_SWATCH_SYMBOL, g])
        row_colors.append(group_color.get(g, "black"))

    _create_legend_table(table_ax, rows, row_colors)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -------------------------------------------------------------------------------------------
# Main Export Function
# -------------------------------------------------------------------------------------------
def export_latent_eval_to_pdf(cfg: Arguments) -> None:
    """Export latent_size hyperparameter comparison plots to PDFs.

    One PDF is created per metric, showing how that metric varies with latent_size
    across all runs, grouped by top-level folder.

    Parameters
    ----------
    cfg : Arguments
        Configuration specifying directories and metric reduction mode.

    Notes
    -----
    - Runs are grouped by their top-level folder in the log directory.
    - Only runs with both hparams.yaml and the metric are included.
    - One PDF is generated per metric tag found across all runs.
    """
    _configure_matplotlib()

    root = Path(cfg.log_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()

    # Validate input directory
    if not root.exists():
        print(f"[gen_latenteval] log_dir does not exist: {root}", file=sys.stderr)
        return

    # Discover run directories
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"[gen_latenteval] no TensorBoard runs found in: {root}", file=sys.stderr)
        return

    # Load per-run: name, hparams, and all metric tags
    run_names: List[str] = []
    run_hparams: List[Dict[str, Any]] = []
    run_tags: List[List[str]] = []

    for run_dir in run_dirs:
        run_name = str(run_dir.relative_to(root))
        hp_file = _find_hparams_file(run_dir)
        if hp_file is None:
            continue
        try:
            hps = _load_hparams(hp_file)
        except Exception as exc:
            print(f"[gen_latenteval] failed to read {hp_file}: {exc}", file=sys.stderr)
            continue

        tags = _load_all_metrics(run_dir)
        if not tags:
            continue

        run_names.append(run_name)
        run_hparams.append(hps)
        run_tags.append(tags)

    if not run_names:
        print("[gen_latenteval] no runs with hparams.yaml and metrics found", file=sys.stderr)
        return

    # Collect union of all metric tags
    all_tags: set[str] = set()
    for tags in run_tags:
        all_tags.update(tags)

    if not all_tags:
        print("[gen_latenteval] no metric tags found", file=sys.stderr)
        return

    # Group runs by top-level folder and assign one color per group
    groups: Dict[str, List[str]] = {}
    for rn in run_names:
        groups.setdefault(_top_level_folder(rn), []).append(rn)
    group_color = _build_group_color_map(list(groups.keys()))

    # For each metric tag, load values and create a plot
    for tag in sorted(all_tags):
        # Load metric value for each run
        runs_with_metric: Dict[str, Tuple[Dict[str, Any], float]] = {}

        for run_dir, run_name, hparams in zip(run_dirs, run_names, run_hparams):
            metric_val = _load_metric_value(run_dir, tag, cfg.metric_reduce)
            if metric_val is None:
                continue
            runs_with_metric[run_name] = (hparams, metric_val)

        if not runs_with_metric:
            continue

        # Create one PDF for this metric
        tag_safe = tag.replace("/", "_")
        out_path = out_dir / f"{tag_safe}.pdf"
        _plot_metric_by_latent_size(tag, runs_with_metric, group_color, out_path)
        print(f"[gen_latenteval] exported: {out_path}")


# -------------------------------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """CLI entrypoint for latent_size hyperparameter comparison export.

    Example
    -------
    Export latent_size comparison plots for all metrics::

        python -m figures.gen_latenteval --log_dir logs --out_dir figures/latenteval \\
            --metric_reduce last
    """
    cfg = Arguments()
    export_latent_eval_to_pdf(cfg)
