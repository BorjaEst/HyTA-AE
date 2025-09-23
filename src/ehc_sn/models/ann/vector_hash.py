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
    def __init__(self, scales: list[int]):
        self._s = torch.tensor(scales, dtype=torch.long)
        self._strides = grid_tools.compute_strides(self._s)
        self.period = int(torch.prod(self._s).item())

    # -----------------------------------------------------------------------------------
    def encode(self, position: tuple[int, int]) -> list[torch.Tensor]:
        rs = grid_tools.extract_digits(position[0], self._strides, self._s)
        cs = grid_tools.extract_digits(position[1], self._strides, self._s)
        idx = rs * self._s + cs
        return [grid_tools.create_grid(idx[i], self._s[i]) for i in range(len(self._s))]

    # -----------------------------------------------------------------------------------
    def decode(self, grids: list[torch.Tensor]) -> tuple[int, int]:
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
    def __init__(self, contexts: list[int]):
        self._c = torch.tensor(contexts, dtype=torch.long)

    # -----------------------------------------------------------------------------------
    def encode(self, context: list[int]) -> list[Tensor]:
        seq = enumerate(torch.tensor(context, dtype=torch.long))
        return [one_hot(ctx, num_classes=self._c[i].item()) for i, ctx in seq]

    # -----------------------------------------------------------------------------------
    def decode(self, contexts: list[Tensor]) -> list[int]:
        return [int(torch.argmax(ctx).item()) for ctx in contexts]

    # -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._c)


if __name__ == "__main__":
    mec = MECLayerII(scales=[2, 3, 6])
    lec = LECLayerII(contexts=[4, 5, 3])
    positions: list[tuple[int, int]] = [(0, 0), (17, 5), (35, 35)]
    context_inputs: list[list[int]] = [[0, 2, 1], [3, 4, 0], [1, 1, 2]]

    print("MECLayerII demo")
    print(f"scales={mec._s.tolist()}, period={mec.period}")
    for i, s in enumerate(mec._s):
        print(f"  scale={s.item()} -> grid shape=({s.item()}, {s.item()})")

    print(f"\nLECLayerII demo")
    print(f"contexts={lec._c.tolist()}")
    for i, c in enumerate(lec._c):
        print(f"  context[{i}] size={c.item()}")

    for pos, ctx in zip(positions, context_inputs):
        grids = mec.encode(pos)
        cells = [grid_tools.extract_cell_coords(g) for g in grids]
        decoded = mec.decode(grids)
        re_cells = [grid_tools.extract_cell_coords(g) for g in mec.encode(decoded)]

        ctx_tensors = lec.encode(ctx)
        decoded_ctx = lec.decode(ctx_tensors)
        re_ctx = lec.decode(lec.encode(decoded_ctx))

        print(f"\nposition={pos}, context={ctx}")
        print(f"  MEC active cells per scale: {cells}")
        print(f"  MEC decoded position: {decoded}")
        print(f"  MEC re-encoded cells: {re_cells}")
        print(f"  LEC decoded context: {decoded_ctx}")
        print(f"  LEC re-encoded context: {re_ctx}")
