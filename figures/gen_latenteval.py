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

import matplotlib.pyplot as plt
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
    extract_top_level_folder,
    fit_table_to_full_width,
    format_float,
)

try:  # YAML is used to read hyperparameters
    import yaml
except Exception:  # pragma: no cover - surfaced at runtime with clear error
    yaml = None  # type: ignore[assignment]


# -------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------
HYPERPARAMETER_KEY = "latent_size"  # The hyperparameter to plot on x-axis
DEFAULT_LEGEND_HPARAMS = ["layer2_size", "sparsity_lambda"]  # Hparams to display in legend
GROUP_COLUMN_INDEX = 1


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

    # Legend hyperparameters
    legend_hparams: List[str] = Field(
        default_factory=lambda: DEFAULT_LEGEND_HPARAMS.copy(),
        description="Hyperparameters to display in legend and use for grouping",
    )


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
# Grouping & Color Mapping
# -------------------------------------------------------------------------------------------
def _extract_folder_from_group_key(group_key: str) -> str:
    """Extract folder from a group key like 'folder|hp1=val1|hp2=val2'.

    Parameters
    ----------
    group_key : str
        Group key containing folder and hyperparameters.

    Returns
    -------
    str
        The folder component.
    """
    parts = group_key.split("|", 1)
    return parts[0] if parts else group_key


def _build_group_key(run_name: str, hparams: Dict[str, Any], legend_hparams: List[str]) -> str:
    """Build a group key from folder and legend hyperparameters.

    Parameters
    ----------
    run_name : str
        Run identifier (relative path).
    hparams : Dict[str, Any]
        Hyperparameters for this run.
    legend_hparams : List[str]
        Hyperparameters to include in grouping.

    Returns
    -------
    str
        Group key like 'folder|hp1=val1|hp2=val2'.
    """
    folder = extract_top_level_folder(run_name)
    hp_parts = []
    for hp in sorted(legend_hparams):
        val = hparams.get(hp, "N/A")
        hp_parts.append(f"{hp}={val}")
    if hp_parts:
        return f"{folder}|{'|'.join(hp_parts)}"
    return folder


# -------------------------------------------------------------------------------------------
# Figure Creation
# -------------------------------------------------------------------------------------------
def _create_legend_table(
    table_axes,
    rows: List[List[str]],
    row_colors: List[str],
    column_labels: List[str],
) -> None:
    """Create and style a legend table under a plot.

    Parameters
    ----------
    table_axes
        Axes for the legend table.
    rows : List[List[str]]
        Table rows with text (swatch column should contain COLOR_SWATCH_SYMBOL).
    row_colors : List[str]
        Colors for each row's swatch.
    column_labels : List[str]
        Column header labels.
    """
    if not rows:
        return

    table = table_axes.table(
        cellText=rows,
        colLabels=column_labels,
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Apply column width fitting
    fit_table_to_full_width(
        table,
        ncols=len(column_labels),
        swatch_col=SWATCH_COLUMN_INDEX,
        expandable_col=GROUP_COLUMN_INDEX,
        total_width=TABLE_WIDTH,
        min_swatch=SWATCH_MIN_WIDTH,
    )

    # Apply cell styles
    apply_table_cell_styles(table, len(column_labels), len(rows))

    # Apply swatch colors
    apply_swatch_colors(table, row_colors)

    table_axes.axis("off")


# -------------------------------------------------------------------------------------------
# PDF Export Functions
# -------------------------------------------------------------------------------------------
def _plot_metric_by_latent_size(
    metric_tag: str,
    runs_data: Dict[str, Tuple[Dict[str, Any], float]],
    group_color_map: Dict[str, str],
    legend_hparams: List[str],
    out_path: Path,
) -> None:
    """Create one PDF for a metric showing latent_size (x) vs metric (y) per group.

    Parameters
    ----------
    metric_tag : str
        The metric tag to plot.
    runs_data : Dict[str, Tuple[Dict[str, Any], float]]
        Mapping from run_name to (hparams, metric_value).
    group_color_map : Dict[str, str]
        Mapping from group_key to color.
    legend_hparams : List[str]
        Hyperparameter keys to display in legend.
    out_path : Path
        Output PDF file path.
    """
    # Build group keys for each run
    group_key_for_run: Dict[str, str] = {}
    for run_name, (hparams, _) in runs_data.items():
        group_key_for_run[run_name] = _build_group_key(run_name, hparams, legend_hparams)

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

        g = group_key_for_run[run_name]
        group_points.setdefault(g, []).append((latent_size_float, metric_val))

    # Filter out groups with no points
    group_entries = [(g, pts) for g, pts in group_points.items() if pts]
    if not group_entries:
        return

    # Create figure sized to number of groups (legend rows)
    fig, (plot_ax, table_ax) = create_figure_with_table(num_rows=len(group_entries))

    # Draw one line per group: sort by latent_size
    for g, pts in sorted(group_entries, key=lambda it: it[0]):
        pts_sorted = sorted(pts, key=lambda p: p[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        c = group_color_map.get(g, "black")
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

    # Build legend table rows with hyperparameters
    column_labels = ["", "Group"] + legend_hparams
    legend_rows: List[List[str]] = []
    row_colors: List[str] = []

    for g, _pts in sorted(group_entries, key=lambda it: it[0]):
        c = group_color_map.get(g, "black")
        row: List[str] = [COLOR_SWATCH_SYMBOL, g]

        # Extract hparam values from group key
        # Group key format: "folder|hp1=val1|hp2=val2"
        parts = g.split("|")
        hparam_dict: Dict[str, str] = {}
        for part in parts[1:]:  # skip folder part
            if "=" in part:
                k, v = part.split("=", 1)
                hparam_dict[k] = v

        # Add only the values (not "hp=val"), formatted as floats if possible
        for hp_key in legend_hparams:
            val = hparam_dict.get(hp_key, "N/A")
            # Try to format as float for numeric values
            try:
                val_float = float(val)
                row.append(format_float(val_float))
            except (ValueError, TypeError):
                row.append(str(val))

        legend_rows.append(row)
        row_colors.append(c)

    _create_legend_table(table_ax, legend_rows, row_colors, column_labels)

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
    configure_matplotlib()

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

    # Build group keys and color map using folder + legend_hparams
    hparams_dict = {rn: hp for rn, hp in zip(run_names, run_hparams)}
    group_key_for_run = {rn: _build_group_key(rn, hparams_dict[rn], cfg.legend_hparams) for rn in run_names}
    unique_groups = sorted(set(group_key_for_run.values()))
    group_color_map = assign_folder_based_colors(unique_groups, _extract_folder_from_group_key)

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
        _plot_metric_by_latent_size(
            tag,
            runs_with_metric,
            group_color_map,
            cfg.legend_hparams,
            out_path,
        )
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
