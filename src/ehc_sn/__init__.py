"""Entorhinal-Hippocampal Complex (EHC) Spatial Navigation library.

This library provides components for modeling spatial navigation mechanisms
in the entorhinal-hippocampal complex.
"""

from ehc_sn.loss import GramianOrthogonalityLoss, HomeostaticActivityLoss, TargetL1SparsityLoss
from ehc_sn.metrics import MetricsLogger

__all__ = [
    "GramianOrthogonalityLoss",
    "HomeostaticActivityLoss",
    "TargetL1SparsityLoss",
    "MetricsLogger",
]
