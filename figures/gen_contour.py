"""Generate contour plots over two hyperparameters from TensorBoard logs.

This module scans TensorBoard log directories, reads hyperparameters from
``hparams.yaml`` files, extracts scalar metrics per run, and generates
publication-ready filled contour plots showing how a metric varies as a
function of two selected hyperparameters (defaults: ``latent_size`` and
``layer2_size``).

One PDF is created per scalar metric tag discovered across runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensorboard.backend.event_processing import event_accumulator

from ehc_sn.utils.figures import assign_folder_based_colors, configure_matplotlib, extract_top_level_folder

try:  # Optional dependency for reading YAML hparams
    import yaml  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    yaml = None


# -------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------
DEFAULT_X_HPARAM = "latent_size"
DEFAULT_Y_HPARAM = "layer2_size"


# -------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------
class Arguments(BaseSettings):
    """Configuration for 2D hyperparameter contour export to PDF.

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
    metric_reduction : str
        How to reduce metric values within the step range: ``last`` (default),
        ``min``, ``max``, or ``mean``.
    hparam_x : str
        Name of the hyperparameter to plot on the x-axis.
    hparam_y : str
        Name of the hyperparameter to plot on the y-axis.
    levels : int
        Number of contour levels to draw.
    cmap : str
        Matplotlib colormap name for filled contours.
    overlay_scatter : bool
        Whether to overlay run points as a scatter and color by folder.
    annotate_points : bool
        Whether to annotate each point with its metric value.
    vmin : Optional[float]
        Fixed minimum for color scale. If None, inferred from data.
    vmax : Optional[float]
        Fixed maximum for color scale. If None, inferred from data.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # IO settings
    log_dir: str = Field(default="logs", description="Directory with experiment logs to parse")
    out_dir: str = Field(default="figures/contours", description="Directory for output PDFs")

    # Metric extraction
    step_start: Optional[int] = Field(
        default=None, description="Starting step for metric extraction (None=first available step)"
    )
    step_end: Optional[int] = Field(
        default=None, description="Ending step for metric extraction (None=last available step)"
    )
    metric_reduction: str = Field(default="last", description="Reduction method over step range: last|min|max|mean")

    # Hyperparameters for axes
    hparam_x: str = Field(default=DEFAULT_X_HPARAM, description="Hyperparameter on x-axis")
    hparam_y: str = Field(default=DEFAULT_Y_HPARAM, description="Hyperparameter on y-axis")

    # Plot styling
    levels: int = Field(default=12, ge=3, description="Number of contour levels")
    cmap: str = Field(default="viridis", description="Matplotlib colormap for filled contours")
    overlay_scatter: bool = Field(default=True, description="Overlay runs as scatter points")
    annotate_points: bool = Field(default=False, description="Annotate each point with metric value")
    vmin: Optional[float] = Field(default=None, description="Lower bound for color scale")
    vmax: Optional[float] = Field(default=None, description="Upper bound for color scale")


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


def _load_metric_value(
    run_dir: Path,
    tag: str,
    step_start: Optional[int] = None,
    step_end: Optional[int] = None,
    reduction: str = "last",
) -> Optional[float]:
    """Load a scalar series for ``tag`` and extract a single value from step range.

    Returns None if tag not found or no values in the requested range.
    """
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

    # Filter steps by requested range (inclusive)
    filtered: List[float] = []
    for s in scalars:
        step = int(s.step)
        if step_start is not None and step < step_start:
            continue
        if step_end is not None and step > step_end:
            continue
        filtered.append(float(s.value))

    if not filtered:
        return None

    reduction = (reduction or "last").lower()
    if reduction == "min":
        return min(filtered)
    if reduction == "max":
        return max(filtered)
    if reduction == "mean":
        return float(sum(filtered)) / float(len(filtered))
    return filtered[-1]


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
# Data Aggregation
# -------------------------------------------------------------------------------------------
def _coerce_float(val: Any) -> Optional[float]:
    """Try to coerce a value to float, returning None on failure."""
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _group_duplicates(
    xs: Iterable[float], ys: Iterable[float], zs: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group duplicate (x, y) points by averaging ``z`` values.

    This helps when multiple runs share identical hyperparameter pairs.
    """
    coords = {}
    for x, y, z in zip(xs, ys, zs):
        key = (float(x), float(y))
        if key not in coords:
            coords[key] = [z]
        else:
            coords[key].append(z)
    out_x, out_y, out_z = [], [], []
    for (x, y), vals in coords.items():
        out_x.append(x)
        out_y.append(y)
        out_z.append(float(np.mean(vals)))
    return np.asarray(out_x), np.asarray(out_y), np.asarray(out_z)


# -------------------------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------------------------
def _plot_contour_for_metric(
    tag: str,
    points: List[Tuple[float, float, float, str]],
    out_path: Path,
    cfg: Arguments,
) -> None:
    """Create one PDF for a metric showing z(tag) over (hparam_x, hparam_y).

    Parameters
    ----------
    tag : str
        The metric tag to plot.
    points : List[Tuple[float, float, float, str]]
        Sequence of (x, y, z, run_name) triples.
    out_path : Path
        Output PDF file path.
    cfg : Arguments
        Configuration object with plotting options.
    """
    if not points:
        return

    # Unpack and de-duplicate identical (x, y) by averaging z
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    run_names = [p[3] for p in points]

    X, Y, Z = _group_duplicates(xs, ys, zs)

    if len(Z) < 3:  # Need at least a triangle
        print(f"[gen_contour] insufficient points for tricontourf: {len(Z)} for tag={tag}", file=sys.stderr)
        return

    # Determine color limits
    vmin = float(cfg.vmin) if cfg.vmin is not None else float(np.nanmin(Z))
    vmax = float(cfg.vmax) if cfg.vmax is not None else float(np.nanmax(Z))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = None, None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Simple 1-axes figure tailored for contours
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    tri = Triangulation(X, Y)
    cf = ax.tricontourf(tri, Z, levels=int(cfg.levels), cmap=cfg.cmap, vmin=vmin, vmax=vmax)
    ax.tricontour(tri, Z, levels=int(cfg.levels), colors="k", linewidths=0.35, alpha=0.55)
    cbar = fig.colorbar(cf, ax=ax, label=tag.split("/")[-1])

    ax.set_xlabel(cfg.hparam_x)
    ax.set_ylabel(cfg.hparam_y)
    ax.set_title(tag)

    # Optional scatter overlay colored by folder for quick provenance
    if cfg.overlay_scatter:
        group_color = assign_folder_based_colors([extract_top_level_folder(rn) for rn in run_names])
        colors = [group_color.get(extract_top_level_folder(rn), "#000000") for rn in run_names]
        ax.scatter(xs, ys, c=colors, s=20, edgecolors="white", linewidths=0.4)

        if cfg.annotate_points:
            for x, y, z in zip(xs, ys, zs):
                try:
                    ax.annotate(f"{z:.3g}", (x, y), textcoords="offset points", xytext=(3, 2), fontsize=8)
                except Exception:
                    pass

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# -------------------------------------------------------------------------------------------
# Main Export Function
# -------------------------------------------------------------------------------------------
def export_hp_contours_to_pdf(cfg: Arguments) -> None:
    """Export 2D hyperparameter contour plots to PDFs.

    One PDF is created per metric, showing how that metric varies with
    ``cfg.hparam_x`` and ``cfg.hparam_y`` across all runs found under
    ``cfg.log_dir``.
    """
    configure_matplotlib()

    root = Path(cfg.log_dir).expanduser().resolve()
    out_dir = Path(cfg.out_dir).expanduser().resolve()

    # Validate input directory
    if not root.exists():
        print(f"[gen_contour] log_dir does not exist: {root}", file=sys.stderr)
        return

    # Discover run directories
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"[gen_contour] no TensorBoard runs found in: {root}", file=sys.stderr)
        return

    # Collect per-run metadata and available tags
    run_names: List[str] = []
    run_hparams: List[Dict[str, Any]] = []
    run_tags: List[List[str]] = []
    kept_run_dirs: List[Path] = []

    for run_dir in run_dirs:
        run_name = str(run_dir.relative_to(root))
        hp_file = _find_hparams_file(run_dir)
        if hp_file is None:
            # No hyperparameters file; skip
            continue
        try:
            hps = _load_hparams(hp_file)
        except Exception as exc:
            print(f"[gen_contour] failed to load hparams for {run_dir}: {exc}", file=sys.stderr)
            continue

        # Require both axes hyperparameters to exist
        if cfg.hparam_x not in hps or cfg.hparam_y not in hps:
            continue

        tags = _load_all_metrics(run_dir)
        if not tags:
            continue

        run_names.append(run_name)
        run_hparams.append(hps)
        run_tags.append(tags)
        kept_run_dirs.append(run_dir)

    if not run_names:
        print("[gen_contour] no runs with required hparams and metrics found", file=sys.stderr)
        return

    # Collect union of all metric tags
    all_tags: set[str] = set()
    for tags in run_tags:
        all_tags.update(tags)

    if not all_tags:
        print("[gen_contour] no metric tags found", file=sys.stderr)
        return

    # For each metric tag, collect (x, y, z) across runs and draw a contour
    for tag in sorted(all_tags):
        points: List[Tuple[float, float, float, str]] = []  # (x, y, z, run_name)
        for run_dir, run_name, hps, tags in zip(kept_run_dirs, run_names, run_hparams, run_tags):
            if tag not in tags:
                continue
            x = _coerce_float(hps.get(cfg.hparam_x, None))
            y = _coerce_float(hps.get(cfg.hparam_y, None))
            if x is None or y is None:
                continue

            z = _load_metric_value(
                run_dir,
                tag,
                step_start=cfg.step_start,
                step_end=cfg.step_end,
                reduction=cfg.metric_reduction,
            )
            if z is None or not np.isfinite(z):
                continue
            points.append((x, y, float(z), run_name))

        if not points:
            # Nothing to plot for this tag
            continue

        tag_safe = tag.replace("/", "_")
        out_path = out_dir / f"{tag_safe}_contour.pdf"
        _plot_contour_for_metric(tag, points, out_path, cfg)


# -------------------------------------------------------------------------------------------
# CLI Entrypoint
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """CLI entrypoint for 2D hyperparameter contour export.

    Examples
    --------
    Export using last value from all steps (default)::

        python -m figures.gen_contour --log_dir logs --out_dir figures/contours

    Export using minimum value from steps 1000 to 5000 with custom axes::

        python -m figures.gen_contour --step_start 1000 --step_end 5000 \
            --metric_reduction min --hparam_x latent_size --hparam_y layer2_size

    """
    cfg = Arguments()
    export_hp_contours_to_pdf(cfg)
