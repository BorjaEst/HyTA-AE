import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, flatten, nn, unflatten
from torch.nn.functional import one_hot
from torch.optim import Adam, Optimizer

from ehc_sn.core import ann
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.modules import dfa, htl
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss
from ehc_sn.utils import grid_tools


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Dentate Gyrus parameters
    latent_size: int = Field(256, gt=0, description="Dimensionality of the DG latent space.")

    # Toroidal manifold shapes
    grid_sizes: List[int] = Field([2, 3, 6], description="List of grid scales.")
    contexts_size: int = Field([30] * 4, description="List of context vector sizes.")

    # HPC CA3 parameters
    states_size: int = Field(32, gt=0, description="Number of clusters each CA3 cluster.")
    n_states: int = Field(16, gt=0, description="Number of CA3 clusters.")


# -------------------------------------------------------------------------------------------
class MECLayerII:
    def __init__(self, scales: List[int]):
        self._s = torch.tensor(scales, dtype=torch.long)
        self._strides = grid_tools.compute_strides(self._s)
        self.period = int(torch.prod(self._s).item())

    # -----------------------------------------------------------------------------------
    def encode(self, position: Tuple[int, int]) -> List[torch.Tensor]:
        rs = grid_tools.extract_digits(position[0], self._strides, self._s)
        cs = grid_tools.extract_digits(position[1], self._strides, self._s)
        idx = rs * self._s + cs
        return [grid_tools.create_grid(idx[i], self._s[i]) for i in range(len(self._s))]

    # -----------------------------------------------------------------------------------
    def decode(self, grids: List[torch.Tensor]) -> Tuple[int, int]:
        coords = [grid_tools.extract_cell_coords(g) for g in grids]
        r_digits = torch.tensor([r for r, _ in coords], dtype=torch.long)
        c_digits = torch.tensor([c for _, c in coords], dtype=torch.long)
        r = int((r_digits * self._strides).sum().item())
        c = int((c_digits * self._strides).sum().item())
        return r, c

    # -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._s)


class LECLayerII:
    def __init__(self, contexts: List[int]):
        self._c = torch.tensor(contexts, dtype=torch.long)

    # -----------------------------------------------------------------------------------
    def encode(self, context: List[int]) -> List[Tensor]:
        seq = enumerate(torch.tensor(context, dtype=torch.long))
        return [one_hot(ctx, num_classes=self._c[i].item()) for i, ctx in seq]

    # -----------------------------------------------------------------------------------
    def decode(self, contexts: List[Tensor]) -> List[int]:
        return [int(torch.argmax(ctx).item()) for ctx in contexts]

    # -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._c)


class VectorHaSH(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()
        self.params = params
        self.mec_layerII = MECLayerII(scales=params.grid_sizes)
        self.lec_layerII = LECLayerII(contexts=params.contexts_size)

    # -----------------------------------------------------------------------------------
    def forward(self, position: Tuple[int, int], context: List[int]) -> None:
        mec_targets = self.mec_layerII.encode(position)  # One-hot grids per scale
        lec_targets = self.lec_layerII.encode(context)  # One-hot vectors per context
        # Further processing can be added here
        return mec_targets, lec_targets


if __name__ == "__main__":
    # Test hippocampal state generation with position and context
    print("=== Testing VectorHaSH with position and context ===")

    # Create VectorHaSH model with parameters
    params = ModelParams(grid_sizes=[2, 3, 6], contexts_size=[4, 5, 3])
    model = VectorHaSH(params)

    positions: List[Tuple[int, int]] = [(0, 0), (17, 5), (35, 35)]
    context_inputs: List[List[int]] = [[0, 2, 1], [3, 4, 0], [1, 1, 2]]

    print("VectorHaSH demo")
    print(f"MEC scales={model.mec_layerII._s.tolist()}, period={model.mec_layerII.period}")
    for i, s in enumerate(model.mec_layerII._s):
        print(f"  scale={s.item()} -> grid shape=({s.item()}, {s.item()})")

    print(f"\nLEC contexts={model.lec_layerII._c.tolist()}")
    for i, c in enumerate(model.lec_layerII._c):
        print(f"  context[{i}] size={c.item()}")

    for pos, ctx in zip(positions, context_inputs):
        # Use VectorHaSH forward method
        mec_targets, lec_targets = model.forward(pos, ctx)

        # Extract information for display
        mec_cells = [grid_tools.extract_cell_coords(g) for g in mec_targets]
        mec_decoded = model.mec_layerII.decode(mec_targets)
        mec_re_cells = [grid_tools.extract_cell_coords(g) for g in model.mec_layerII.encode(mec_decoded)]

        lec_decoded = model.lec_layerII.decode(lec_targets)
        lec_re_encoded = model.lec_layerII.decode(model.lec_layerII.encode(lec_decoded))

        print(f"\nposition={pos}, context={ctx}")
        print(f"  MEC active cells per scale: {mec_cells}")
        print(f"  MEC decoded position: {mec_decoded}")
        print(f"  MEC re-encoded cells: {mec_re_cells}")
        print(f"  LEC decoded context: {lec_decoded}")
        print(f"  LEC re-encoded context: {lec_re_encoded}")
