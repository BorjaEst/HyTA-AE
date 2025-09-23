"""
Encoding utilities for entorhinal cortex layers.

This module provides shared encoding/decoding utilities for both:
- MECLayerII: Grid cell encoding with mixed-radix positional representations
- LECLayerII: Categorical context encoding with one-hot representations

The module centralizes common patterns like batched tensor operations,
one-hot encoding/decoding, and maintains backward compatibility with
the original grid_tools interface.
"""

from typing import List, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import one_hot

# -------------------------------------------------------------------------------------------
# Shared encoding utilities for both MEC and LEC layers
# -------------------------------------------------------------------------------------------


def encode_categorical_batch(contexts: Tensor, num_classes: Tensor) -> List[Tensor]:
    """
    Encode batch of categorical contexts to one-hot representations.

    Parameters
    ----------
    contexts : Tensor
        Tensor of shape (B, K) with categorical indices
    num_classes : Tensor
        Tensor of shape (K,) with number of classes for each dimension

    Returns
    -------
    List[Tensor]
        List over K dimensions with tensors of shape (B, C_i)
    """
    B, K = contexts.shape
    return [one_hot(contexts[:, i], num_classes=int(num_classes[i].item())).float() for i in range(K)]


def decode_categorical_batch(encodings: List[Tensor]) -> Tensor:
    """
    Decode batch of one-hot categorical encodings to indices.

    Parameters
    ----------
    encodings : List[Tensor]
        List over K dimensions with tensors of shape (B, C_i)

    Returns
    -------
    Tensor
        Tensor of shape (B, K) with categorical indices
    """
    return torch.stack([torch.argmax(C, dim=-1) for C in encodings], dim=-1)


# -------------------------------------------------------------------------------------------
# Grid-specific utilities for MEC layers
# -------------------------------------------------------------------------------------------


def compute_strides(scales: Tensor) -> Tensor:
    """Compute cumulative strides for mixed-radix encoding."""
    return torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), scales[:-1]]), dim=0)


def extract_grid_digits(positions: Tensor, strides: Tensor, scales: Tensor, period: int) -> Tensor:
    """
    Extract mixed-radix digits from batch of positions using vectorized operations.

    Parameters
    ----------
    positions : Tensor
        Positions tensor of shape (B, 2) with (row, col) coordinates
    strides : Tensor
        Cumulative strides for mixed-radix encoding (S,)
    scales : Tensor
        Scale sizes (S,)
    period : int
        Torus period for wrapping coordinates

    Returns
    -------
    Tensor
        Digit indices of shape (B, S) for flattened grid positions
    """
    r = positions[:, 0].long() % period
    c = positions[:, 1].long() % period

    # Base-s decomposition for all scales at once
    r_digits = (r.unsqueeze(1) // strides) % scales  # (B, S)
    c_digits = (c.unsqueeze(1) // strides) % scales  # (B, S)
    return r_digits * scales + c_digits  # (B, S)


def encode_grid_batch(indices: Tensor, scales: Tensor) -> List[Tensor]:
    """
    Create batch of one-hot grids from digit indices.

    Parameters
    ----------
    indices : Tensor
        Digit indices of shape (B, S)
    scales : Tensor
        Scale sizes (S,)

    Returns
    -------
    List[Tensor]
        List over scales with tensors of shape (B, s_i*s_i)
    """
    return [one_hot(indices[:, i], num_classes=int(scales[i].item() ** 2)).float() for i in range(len(scales))]


def decode_grid_batch(grids: List[Tensor], scales: Tensor, strides: Tensor) -> Tensor:
    """
    Extract coordinates from batch of grid activations.

    Parameters
    ----------
    grids : List[Tensor]
        List over scales with tensors of shape (B, s_i*s_i)
    scales : Tensor
        Scale sizes (S,)
    strides : Tensor
        Cumulative strides (S,)

    Returns
    -------
    Tensor
        Tensor of shape (B, 2) with (row, col) coordinates
    """
    B = grids[0].shape[0]
    r_acc = torch.zeros(B, dtype=torch.long)
    c_acc = torch.zeros(B, dtype=torch.long)

    for i, G in enumerate(grids):
        s_i = scales[i].item()
        idx = torch.argmax(G, dim=-1)  # (B,)
        r_digit = idx // s_i
        c_digit = idx % s_i
        r_acc += r_digit * strides[i]
        c_acc += c_digit * strides[i]

    return torch.stack([r_acc, c_acc], dim=-1)  # (B, 2)
