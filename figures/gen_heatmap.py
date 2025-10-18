"""Generate time-series heatmaps showing metric evolution over training steps and hyperparameters.

This module scans TensorBoard log directories, reads hyperparameters from
``hparams.yaml`` files, extracts scalar metric time series per run, and generates
publication-ready heatmaps using ``pcolormesh`` with grid interpolation. This allows
visualizing how a metric evolves during training across different hyperparameter values,
even when measurements are sparse or irregular.

One PDF is created per scalar metric tag discovered across runs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.interpolate import griddata
from tensorboard.backend.event_processing import event_accumulator

from ehc_sn.utils.figures import configure_matplotlib

# -------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------
DEFAULT_Y_HPARAM = "latent_dim"


# -------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------
class Arguments(BaseSettings):
    """Configuration for training step × hyperparameter heatmap export to PDF.

    Attributes
    ----------
    log_dir : str
        Directory containing TensorBoard experiment logs.
    out_dir : str
        Directory where output PDFs will be saved.
    step_start : Optional[int]
        Starting step for metric extraction. If None, uses first available step.
    step_end : Optional[int]
        Ending step for metric extraction. If None, uses last available step.
    step_interval : Optional[int]
        Sampling interval for steps. If None, uses all available steps.
    hparam_y : str
        Name of the hyperparameter to plot on the y-axis.
    grid_resolution : int
        Number of grid points for interpolation in each dimension.
    interpolation_method : str
        Interpolation method for griddata: ``linear``, ``nearest``, or ``cubic``.
    cmap : str
        Matplotlib colormap name for the heatmap.
    vmin : Optional[float]
        Fixed minimum for color scale. If None, inferred from data.
    vmax : Optional[float]
        Fixed maximum for color scale. If None, inferred from data.
    log_y : bool
        Use logarithmic scale for y-axis (hyperparameter).
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # IO settings
    log_dir: str = Field(default="logs", description="Directory with experiment logs to parse")
    out_dir: str = Field(default="figures/heatmaps", description="Directory for output PDFs")

    # Metric extraction
    step_start: Optional[int] = Field(default=None, description="Starting step for metric extraction")
    step_end: Optional[int] = Field(default=None, description="Ending step for metric extraction")
    step_interval: Optional[int] = Field(default=None, description="Sampling interval for steps")

    # Hyperparameter for y-axis
    hparam_y: str = Field(default=DEFAULT_Y_HPARAM, description="Hyperparameter on y-axis")

    # Interpolation and plotting
    grid_resolution: int = Field(default=100, ge=10, description="Grid resolution for interpolation")
    interpolation_method: str = Field(default="linear", description="Interpolation method: linear|nearest|cubic")
    cmap: str = Field(default="viridis", description="Matplotlib colormap for heatmap")
    vmin: Optional[float] = Field(default=None, description="Lower bound for color scale")
    vmax: Optional[float] = Field(default=None, description="Upper bound for color scale")

    # Axis scaling
    log_y: bool = Field(default=False, description="Use logarithmic scale for y-axis (hparam_y)")


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
            cand = cur / name
            if cand.exists():
                return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _flatten_hparams(data: Any) -> Dict[str, Any]:
    """Flatten a simple nested dict of hyperparameters.

    This flattens one or two-level mappings like {"model": {"lr": 1e-3}} to
    {"model.lr": 1e-3}. Lists are joined as comma strings. Non-mappings are
    returned as-is when possible.
    """
    flat: Dict[str, Any] = {}

    def _rec(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _rec(key, v)
        elif isinstance(obj, (list, tuple)):
            flat[prefix] = ",".join(str(x) for x in obj)
        else:
            flat[prefix] = obj

    if isinstance(data, dict):
        _rec("", data)
    return flat


def _load_hparams(hp_file: Path) -> Dict[str, Any]:
    """Load hyperparameters from a YAML file into a flat dict."""
    if yaml is None:
        raise RuntimeError("PyYAML is not available. Please `pip install pyyaml`.")
    with hp_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Prefer top-level simple mappings; fallback to flattened mapping
    if isinstance(data, dict):
        # If it's already flat enough, use as-is; else flatten
        simple = {}
        complex_found = False
        for k, v in data.items():
            if isinstance(v, (dict, list, tuple)):
                complex_found = True
                break
            simple[k] = v
        return simple if not complex_found else _flatten_hparams(data)
    return {}


def _load_metric_timeseries(
    run_dir: Path,
    tag: str,
    step_start: Optional[int] = None,
    step_end: Optional[int] = None,
    step_interval: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Load a scalar series for ``tag`` and return (step, value) pairs within range.

    Parameters
    ----------
    run_dir : Path
        TensorBoard run directory.
    tag : str
        Metric tag to load.
    step_start : Optional[int]
        Starting step (inclusive).
    step_end : Optional[int]
        Ending step (inclusive).
    step_interval : Optional[int]
        Sampling interval for steps.

    Returns
    -------
    List[Tuple[int, float]]
        List of (step, value) tuples.
    """
    if event_accumulator is None:
        raise RuntimeError("tensorboard is not available. Please `pip install tensorboard`.")

    ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return []

    if tag not in ea.Tags().get("scalars", []):
        return []

    scalars = ea.Scalars(tag)
    if not scalars:
        return []

    # Filter and sample steps
    result: List[Tuple[int, float]] = []
    for s in scalars:
        step = int(s.step)
        if step_start is not None and step < step_start:
            continue
        if step_end is not None and step > step_end:
            continue
        if step_interval is not None and step % step_interval != 0:
            continue
        result.append((step, float(s.value)))

    return result


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
# Data Aggregation & Interpolation
# -------------------------------------------------------------------------------------------
def _coerce_float(val: Any) -> Optional[float]:
    """Try to coerce a value to float, returning None on failure."""
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _interpolate_grid(
    steps: np.ndarray,
    hparam_vals: np.ndarray,
    metric_vals: np.ndarray,
    resolution: int,
    method: str,
    log_y: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate sparse (step, hparam, metric) data onto a regular grid.

    Parameters
    ----------
    steps : np.ndarray
        Training step values (x-coordinates).
    hparam_vals : np.ndarray
        Hyperparameter values (y-coordinates).
    metric_vals : np.ndarray
        Metric values (z-values).
    resolution : int
        Number of grid points in each dimension.
    method : str
        Interpolation method: ``linear``, ``nearest``, or ``cubic``.
    log_y : bool
        Whether to use logarithmic spacing for y-axis grid.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Grid arrays: (X_grid, Y_grid, Z_grid).
    """
    # Create regular grid
    x_min, x_max = steps.min(), steps.max()
    y_min, y_max = hparam_vals.min(), hparam_vals.max()

    x_grid = np.linspace(x_min, x_max, resolution)

    if log_y and y_min > 0:
        y_grid = np.logspace(np.log10(y_min), np.log10(y_max), resolution)
    else:
        y_grid = np.linspace(y_min, y_max, resolution)

    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # Interpolate onto grid
    points = np.column_stack([steps, hparam_vals])
    Z_grid = griddata(points, metric_vals, (X_grid, Y_grid), method=method, fill_value=np.nan)

    return X_grid, Y_grid, Z_grid


# -------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------
def _plot_timeseries_heatmap(
    tag: str,
    data: List[Tuple[int, float, float]],  # (step, hparam_y, metric)
    out_path: Path,
    cfg: Arguments,
    step_info: str,
) -> None:
    """Create one PDF for a metric showing evolution over steps and hyperparameter.

    Parameters
    ----------
    tag : str
        The metric tag to plot.
    data : List[Tuple[int, float, float]]
        Sequence of (step, hparam_y, metric_value) tuples.
    out_path : Path
        Output PDF file path.
    cfg : Arguments
        Configuration object with plotting options.
    step_info : str
        Description of step range for title.
    """
    if not data:
        return

    # Unpack data
    steps = np.array([d[0] for d in data], dtype=float)
    hparam_vals = np.array([d[1] for d in data], dtype=float)
    metric_vals = np.array([d[2] for d in data], dtype=float)

    if len(steps) < 3:
        print(
            f"[gen_heatmap] insufficient points for interpolation: {len(steps)} for tag={tag}",
            file=sys.stderr,
        )
        return

    # Interpolate onto regular grid
    try:
        X_grid, Y_grid, Z_grid = _interpolate_grid(
            steps,
            hparam_vals,
            metric_vals,
            resolution=cfg.grid_resolution,
            method=cfg.interpolation_method,
            log_y=cfg.log_y,
        )
    except Exception as exc:
        print(f"[gen_heatmap] interpolation failed for tag={tag}: {exc}", file=sys.stderr)
        return

    # Determine color limits
    vmin = float(cfg.vmin) if cfg.vmin is not None else np.nanmin(Z_grid)
    vmax = float(cfg.vmax) if cfg.vmax is not None else np.nanmax(Z_grid)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Plot heatmap with pcolormesh
    pcm = ax.pcolormesh(X_grid, Y_grid, Z_grid, cmap=cfg.cmap, vmin=vmin, vmax=vmax, shading="auto")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(tag)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(cfg.hparam_y)
    ax.set_title(f"{tag} ({step_info})")

    # Apply axis scales
    if cfg.log_y and np.all(hparam_vals > 0):
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -------------------------------------------------------------------------------------------
# Main Export Function
# -------------------------------------------------------------------------------------------
def export_timeseries_heatmaps_to_pdf(cfg: Arguments) -> None:
    """Export training step × hyperparameter heatmaps to PDFs.

    One PDF is created per metric, showing how that metric evolves over training
    steps across different values of ``cfg.hparam_y``.
    """
    configure_matplotlib()

    root = Path(cfg.log_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()

    # Validate input directory
    if not root.exists():
        print(f"[gen_heatmap] log_dir does not exist: {root}", file=sys.stderr)
        return

    # Discover run directories
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"[gen_heatmap] no TensorBoard runs found in: {root}", file=sys.stderr)
        return

    # Collect per-run metadata and available tags
    run_hparams: List[Dict[str, Any]] = []
    run_tags: List[List[str]] = []
    kept_run_dirs: List[Path] = []

    for run_dir in run_dirs:
        hp_file = _find_hparams_file(run_dir)
        if hp_file is None:
            continue
        try:
            hps = _load_hparams(hp_file)
        except Exception as exc:
            print(f"[gen_heatmap] failed to load hparams for {run_dir}: {exc}", file=sys.stderr)
            continue

        # Require y-axis hyperparameter to exist
        if cfg.hparam_y not in hps:
            continue

        tags = _load_all_metrics(run_dir)
        if not tags:
            continue

        run_hparams.append(hps)
        run_tags.append(tags)
        kept_run_dirs.append(run_dir)

    if not run_hparams:
        print("[gen_heatmap] no runs with required hparams and metrics found", file=sys.stderr)
        return

    # Collect union of all metric tags
    all_tags: set[str] = set()
    for tags in run_tags:
        all_tags.update(tags)

    if not all_tags:
        print("[gen_heatmap] no metric tags found", file=sys.stderr)
        return

    # Build step range info string for titles
    if cfg.step_start is not None and cfg.step_end is not None:
        step_info = f"steps={cfg.step_start}..{cfg.step_end}"
    elif cfg.step_start is not None:
        step_info = f"steps={cfg.step_start}..end"
    elif cfg.step_end is not None:
        step_info = f"steps=start..{cfg.step_end}"
    else:
        step_info = "steps=all"

    if cfg.step_interval is not None:
        step_info += f", interval={cfg.step_interval}"

    # For each metric tag, collect (step, hparam_y, metric) and create heatmap
    for tag in sorted(all_tags):
        data: List[Tuple[int, float, float]] = []  # (step, hparam_y, metric)

        for run_dir, hps, tags in zip(kept_run_dirs, run_hparams, run_tags):
            if tag not in tags:
                continue

            y_val = _coerce_float(hps.get(cfg.hparam_y, None))
            if y_val is None:
                continue

            timeseries = _load_metric_timeseries(
                run_dir,
                tag,
                step_start=cfg.step_start,
                step_end=cfg.step_end,
                step_interval=cfg.step_interval,
            )

            for step, metric_val in timeseries:
                if np.isfinite(metric_val):
                    data.append((int(step), float(y_val), float(metric_val)))

        if not data:
            continue

        tag_safe = tag.replace("/", "_")
        out_path = out_dir / f"{tag_safe}_heatmap.pdf"
        _plot_timeseries_heatmap(tag, data, out_path, cfg, step_info)
        print(f"[gen_heatmap] exported: {out_path}")


# -------------------------------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """CLI entrypoint for training step × hyperparameter heatmap export.

    Examples
    --------
    Export all steps with default hyperparameter::

        python -m figures.gen_heatmap --log_dir logs --out_dir figures/heatmaps

    Export specific step range with custom hyperparameter and interpolation::

        python -m figures.gen_heatmap --step_start 0 --step_end 5000 \
            --step_interval 100 --hparam_y latent_dim --interpolation_method cubic

    Use logarithmic y-axis scale::

        python -m figures.gen_heatmap --log_y --hparam_y learning_rate

    """
    cfg = Arguments()
    export_timeseries_heatmaps_to_pdf(cfg)
