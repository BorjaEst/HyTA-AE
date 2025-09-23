from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor, nn


# -------------------------------------------------------------------------------------------
def compute_strides(scales: Tensor) -> Tensor:
    """Compute cumulative strides for mixed-radix encoding."""
    return torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), scales[:-1]]), dim=0)


# -------------------------------------------------------------------------------------------
def extract_digits(position: int, strides: Tensor, scales: Tensor) -> Tensor:
    """Extract mixed-radix digits from a position value."""
    return (position // strides) % scales


# -------------------------------------------------------------------------------------------
def create_grid(digit: int, scale: Tensor) -> Tensor:
    """Create a one-hot 2D grid for a single digit and scale."""
    digit = torch.tensor(digit, dtype=torch.long)
    grid_flat = nn.functional.one_hot(digit, num_classes=scale * scale)
    return grid_flat.to(torch.float32).reshape(scale, scale)


# -------------------------------------------------------------------------------------------
def extract_cell_coords(grid: Tensor) -> Tuple[int, int]:
    """Extract row, col coordinates from a grid's active cell."""
    idx = int(grid.argmax().item())
    s = grid.shape[0]
    return idx // s, idx % s
