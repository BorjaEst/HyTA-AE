import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, cat, flatten, nn, unflatten
from torch.nn.functional import one_hot
from torch.optim import Adam, Optimizer

from ehc_sn.core import ann
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.modules import dfa, drtp, htl
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss
from ehc_sn.utils import grid_tools


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Dentate Gyrus parameters
    latent_size: int = Field(256, gt=0, description="Dimensionality of the DG latent space.")
    substate_size: int = Field(32, gt=0, description="Dimensionality of each CA3 subcluster.")
    state_loops: int = Field(5, gt=0, description="Number of CA3 recurrent state updates per step.")

    # Toroidal manifold shapes
    grid_sizes: List[int] = Field([2, 3, 6], description="List of grid scales.")
    contexts_size: List[int] = Field([30] * 4, description="List of context vector sizes.")

    @property
    def ec_shapes(self) -> Tuple[List[int], List[int]]:
        return [s**2 for s in self.grid_sizes], self.contexts_size


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


# -------------------------------------------------------------------------------------------
class LECLayerII:
    def __init__(self, contexts: List[int]):
        self._c = torch.tensor(contexts, dtype=torch.long)

    # -----------------------------------------------------------------------------------
    def encode(self, context: List[int]) -> List[Tensor]:
        seq = enumerate(torch.tensor(context, dtype=torch.long))
        return [one_hot(ctx, num_classes=self._c[i].item()).float() for i, ctx in seq]

    # -----------------------------------------------------------------------------------
    def decode(self, contexts: List[Tensor]) -> List[int]:
        return [int(torch.argmax(ctx).item()) for ctx in contexts]

    # -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._c)


# -------------------------------------------------------------------------------------------
class DG(nn.Module):
    def __init__(self, latent_size: int, mec_shape: List[int], lec_shape: List[int]):
        super().__init__()
        syn_mec = nn.Linear(sum(mec_shape), latent_size)
        syn_lec = nn.Linear(sum(lec_shape), latent_size)
        self.synapses = nn.ModuleDict({"mec": syn_mec, "lec": syn_lec})
        self.activation = nn.ReLU()

    def forward(self, mec_inputs: Tensor, lec_inputs: Tensor) -> Tensor:
        x = self.synapses["mec"](mec_inputs) + self.synapses["lec"](lec_inputs)
        return self.activation(x)


# -------------------------------------------------------------------------------------------
class CA3Cluster(nn.Module):
    def __init__(self, latent_units: int, cluster_size: int, target_size: int, ca3_units: int):
        super().__init__()
        self.input_syn = nn.Linear(latent_units, cluster_size)
        self.recurrent_syn = nn.Linear(ca3_units, cluster_size)
        self.activation = nn.ReLU()
        self.target_syn = drtp.Linear(target_size, cluster_size)
        self.state: Optional[Tensor] = None

    # -----------------------------------------------------------------------------------
    def forward(self, dg_input: Tensor, recurrent_input: Tensor) -> Tensor:
        x = self.input_syn(dg_input)
        if recurrent_input is not None:
            x = x + self.recurrent_syn(recurrent_input)
        self.state = self.activation(x)
        return self.target_syn(self.state)  # Store state for feedback

    # -----------------------------------------------------------------------------------
    def feedback(self, target: Tensor) -> None:
        self.target_syn.feedback(target)


# -------------------------------------------------------------------------------------------
class CA3(nn.Module):
    def __init__(self, latent_size: int, cluster_size: int, mec_shape: List[int], lec_shape: List[int]):
        super().__init__()
        self.n_clusters = len(mec_shape) + len(lec_shape)
        self.cluster_size = cluster_size
        mec_clusters = [CA3Cluster(latent_size, cluster_size, x, self.units) for x in mec_shape]
        lec_clusters = [CA3Cluster(latent_size, cluster_size, x, self.units) for x in lec_shape]
        clusters_dict = {"mec": nn.ModuleList(mec_clusters), "lec": nn.ModuleList(lec_clusters)}
        self.clusters = nn.ModuleDict(clusters_dict)

    @property
    def units(self) -> int:
        return self.n_clusters * self.cluster_size

    # -----------------------------------------------------------------------------------
    def _build_recurrent_input(self, batch: int, device: torch.device) -> Tensor:
        parts: List[Tensor] = []
        for c in list(self.clusters["mec"]) + list(self.clusters["lec"]):
            if c.state is None:
                parts.append(torch.zeros(batch, self.cluster_size, device=device))
            else:
                parts.append(c.state)
        return cat(parts, dim=-1)

    # -----------------------------------------------------------------------------------
    def forward(self, dg_pattern: Tensor) -> Tensor:
        squeeze_out = False
        if dg_pattern.ndim == 1:
            dg_pattern = dg_pattern.unsqueeze(0)
            squeeze_out = True
        B = dg_pattern.shape[0]
        device = dg_pattern.device

        recurrent_input = self._build_recurrent_input(B, device)

        outputs: List[Tensor] = []
        for cluster in list(self.clusters["mec"]) + list(self.clusters["lec"]):
            out = cluster(dg_pattern, recurrent_input)
            outputs.append(out)

        y = cat(outputs, dim=-1)
        return y.squeeze(0) if squeeze_out else y

    # -----------------------------------------------------------------------------------
    def feedback_mec(self, targets: List[Tensor]) -> None:
        for i, cluster in enumerate(self.clusters["mec"]):
            cluster.feedback(targets[i].reshape(-1).float())

    def feedback_lec(self, targets: List[Tensor]) -> None:
        for i, cluster in enumerate(self.clusters["lec"]):
            cluster.feedback(targets[i].reshape(-1).float())


# -----------------------------------------------------------------------------------
class VectorHaSH(nn.Module):
    def __init__(self, params: Optional[ModelParams] = None):
        super().__init__()
        self.params = params = params or ModelParams()
        self.mec_layerII = MECLayerII(scales=params.grid_sizes)
        self.lec_layerII = LECLayerII(contexts=params.contexts_size)
        self.dg = DG(params.latent_size, *params.ec_shapes)
        self.ca3 = CA3(params.latent_size, params.substate_size, *params.ec_shapes)

    # -----------------------------------------------------------------------------------
    def forward(self, position: Tuple[int, int], context: List[int]) -> Tensor:
        # Targets from EC
        self.mec_targets = self.mec_layerII.encode(position)  # list[Si x Si]
        self.lec_targets = self.lec_layerII.encode(context)  # list[Ci]

        # Flatten and concatenate EC targets for DG input
        mec_flat = cat([g.reshape(-1) for g in self.mec_targets], dim=0)
        lec_flat = cat([g.reshape(-1) for g in self.lec_targets], dim=0)

        # DG and CA3 dynamics
        dg_pattern = self.dg(mec_flat, lec_flat)
        for _ in range(self.params.state_loops):
            y = self.ca3(dg_pattern)  # recurrent updates use internal cluster states
        return y

    # -----------------------------------------------------------------------------------
    def feedback(self) -> None:
        self.ca3.feedback_mec(self.mec_targets)
        self.ca3.feedback_lec(self.lec_targets)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test hippocampal state generation with a batch of positions and contexts
    print("=== Testing VectorHaSH with batched input ===")

    params = ModelParams(grid_sizes=[2, 3, 6], contexts_size=[4, 5, 3])
    model = VectorHaSH(params)

    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    batch_positions: List[Tuple[int, int]] = [(0, 0), (17, 5), (35, 35)]
    batch_contexts: List[List[int]] = [[0, 2, 1], [3, 4, 0], [1, 1, 2]]

    print("VectorHaSH demo")
    print(f"MEC scales={model.mec_layerII._s.tolist()}, period={model.mec_layerII.period}")
    for s in model.mec_layerII._s:
        print(f"  scale={s.item()} -> grid shape=({s.item()}, {s.item()})")

    print(f"\nLEC contexts={model.lec_layerII._c.tolist()}")
    for i, c in enumerate(model.lec_layerII._c):
        print(f"  context[{i}] size={c.item()}")

    batch_outputs: List[Tensor] = []
    for pos, ctx in zip(batch_positions, batch_contexts):
        # Forward
        y = model.forward(pos, ctx)  # CA3 output for one sample

        # Build EC target vector to match CA3 output
        target_vec = cat(
            [g.reshape(-1).float() for g in (model.mec_targets + model.lec_targets)],
            dim=0,
        )

        # Optimization step (Adam) + DRTP feedback
        optimizer.zero_grad()
        loss = criterion(y, target_vec)
        loss.backward()
        optimizer.step()
        model.feedback()

        batch_outputs.append(y.unsqueeze(0))  # collect as batch

        mec_cells = [grid_tools.extract_cell_coords(g) for g in model.mec_targets]
        mec_decoded = model.mec_layerII.decode(model.mec_targets)
        lec_decoded = model.lec_layerII.decode(model.lec_targets)

        print(f"\nposition={pos}, context={ctx}")
        print(f"  MEC active cells per scale: {mec_cells}")
        print(f"  MEC decoded position: {mec_decoded}")
        print(f"  LEC decoded context: {lec_decoded}")
        print(f"  CA3 sample output shape: {tuple(y.shape)}")
        print(f"  Loss: {loss.item():.6f}")

    Y = torch.cat(batch_outputs, dim=0)
    print(f"\nBatched CA3 output shape: {tuple(Y.shape)}")
