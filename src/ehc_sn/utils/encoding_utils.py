"""
Tensor transformation utilities for neural network operations.

This module provides generic tensor transformation utilities including:
- One-hot encoding/decoding operations for categorical data
- Coordinate transformations and mixed-radix decomposition
- Tensor concatenation and state management utilities
- Neural module application patterns

The utilities are designed to be generic and reusable across different
neural network architectures and domains.
"""

from typing import List, Union

import torch
from torch import Tensor
from torch.nn.functional import one_hot

# -------------------------------------------------------------------------------------------
# One-hot encoding/decoding utilities
# -------------------------------------------------------------------------------------------


def batch_to_onehot(batch_indices: Tensor, num_classes: Tensor) -> List[Tensor]:
    """
    Convert batch of categorical indices to one-hot tensor representations.

    Parameters
    ----------
    batch_indices : Tensor
        Tensor of shape (B, K) with categorical indices
    num_classes : Tensor
        Tensor of shape (K,) with number of classes for each dimension

    Returns
    -------
    List[Tensor]
        List over K dimensions with tensors of shape (B, C_i)
    """
    B, K = batch_indices.shape
    return [one_hot(batch_indices[:, i], num_classes=int(num_classes[i].item())).float() for i in range(K)]


def onehot_to_indices(encodings: List[Tensor]) -> Tensor:
    """
    Convert one-hot tensor encodings back to categorical indices.

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
# Tensor utilities for neural network operations
# -------------------------------------------------------------------------------------------


def create_zero_tensor(batch_size: int, feature_size: int, device: torch.device) -> Tensor:
    """Create a zero tensor with specified dimensions."""
    return torch.zeros(batch_size, feature_size, device=device)


def concat_states(states: List[Tensor], batch_size: int, feature_size: int, device: torch.device) -> Tensor:
    """Concatenate a list of state tensors, replacing None states with zeros."""
    tensors = [
        state.detach() if state is not None else create_zero_tensor(batch_size, feature_size, device)
        for state in states
    ]
    return torch.cat(tensors, dim=-1)


# -------------------------------------------------------------------------------------------
# Coordinate transformation and mixed-radix utilities
# -------------------------------------------------------------------------------------------


def compute_strides(scales: Tensor) -> Tensor:
    """Compute cumulative strides for mixed-radix encoding."""
    return torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), scales[:-1]]), dim=0)


def coordinates_to_indices(positions: Tensor, strides: Tensor, scales: Tensor, period: int) -> Tensor:
    """
    Convert 2D coordinates to flattened indices using mixed-radix decomposition.

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
        Flattened indices of shape (B, S)
    """
    r = positions[:, 0].long() % period
    c = positions[:, 1].long() % period

    # Base-s decomposition for all scales at once
    r_digits = (r.unsqueeze(1) // strides) % scales  # (B, S)
    c_digits = (c.unsqueeze(1) // strides) % scales  # (B, S)
    return r_digits * scales + c_digits  # (B, S)


def indices_to_onehot(indices: Tensor, scales: Tensor) -> List[Tensor]:
    """
    Convert flattened indices to one-hot tensor representations.

    Parameters
    ----------
    indices : Tensor
        Flattened indices of shape (B, S)
    scales : Tensor
        Scale sizes (S,)

    Returns
    -------
    List[Tensor]
        List over scales with tensors of shape (B, s_i*s_i)
    """
    return [one_hot(indices[:, i], num_classes=int(scales[i].item() ** 2)).float() for i in range(len(scales))]


def onehot_to_coordinates(grids: List[Tensor], scales: Tensor, strides: Tensor) -> Tensor:
    """
    Convert one-hot tensor representations back to 2D coordinates.

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
