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
    contexts_size: int = Field([30] * 4, description="List of context vector sizes.")

    @property
    def ec_clusters(self) -> int:
        """Return number of CA3 clusters (one per MEC scale and LEC context)."""
        return len(self.grid_sizes) + len(self.contexts_size)

    @property
    def ec_units(self) -> Tuple[int, int]:
        """Return total number of EC outputs (MEC + LEC)."""
        return (sum(s * s for s in self.grid_sizes), sum(self.contexts_size))


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
        return [one_hot(ctx, num_classes=self._c[i].item()) for i, ctx in seq]

    # -----------------------------------------------------------------------------------
    def decode(self, contexts: List[Tensor]) -> List[int]:
        return [int(torch.argmax(ctx).item()) for ctx in contexts]

    # -----------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._c)


# -------------------------------------------------------------------------------------------
class DG(nn.Module):
    def __init__(self, latent_size: int, mec_units: List[int], lec_units: List[int]):
        super().__init__()
        syn_mec = nn.Linear(sum(mec_units), latent_size)
        syn_lec = nn.Linear(sum(lec_units), latent_size)
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
        self.target_syn = drtp.Linear(cluster_size, target_size)  # TODO this is not correct
        self.activation = nn.ReLU()
        self.state: Optional[Tensor] = None

    # -----------------------------------------------------------------------------------
    def forward(self, dg_input: Tensor, recurrent_input: Optional[Tensor] = None) -> Tensor:
        x = self.input_syn(dg_input)
        if recurrent_input is not None:
            x = x + self.recurrent_syn(recurrent_input)
        self.state = self.activation(x)
        return self.state

    # -----------------------------------------------------------------------------------
    def feedback(self, target: Tensor) -> None:
        return  # TODO


# -------------------------------------------------------------------------------------------
class CA3(nn.Module):
    def __init__(self, latent_size: int, cluster_size: int, mec_units: List[int], lec_units: List[int]):
        super().__init__()
        ca3_units = cluster_size * (len(mec_units) + len(lec_units))
        mec_clusters = [CA3Cluster(latent_size, cluster_size, x, ca3_units) for x in mec_units]
        lec_clusters = [CA3Cluster(latent_size, cluster_size, x, ca3_units) for x in lec_units]
        clusters_dict = {"mec": nn.ModuleList(mec_clusters), "lec": nn.ModuleList(lec_clusters)}
        self.clusters = nn.ModuleDict(clusters_dict)

    # -----------------------------------------------------------------------------------
    @property
    def state(self) -> Optional[Tensor]:
        if self.clusters[0].state is None:  # If 1 None, all None
            return None
        return cat([c.state for c in self.clusters], dim=-1)

    # -----------------------------------------------------------------------------------
    def forward(self, dg_pattern: Tensor) -> Tensor:
        pass  # TODO: Implement recurrent CA3 forward pass

    # -----------------------------------------------------------------------------------
    def set_targets(self, mec_targets: List[Tensor], lec_targets: List[Tensor]) -> None:
        all_targets = mec_targets + lec_targets
        for cluster, target in zip(self.clusters, all_targets):
            cluster.set_target(flatten(target))

    # -----------------------------------------------------------------------------------
    def feedback_mec(self, targets: List[Tensor]) -> None:
        for i, cluster in enumerate(self.clusters["mec"]):
            cluster.feedback(targets[i])

    def feedback_lec(self, targets: List[Tensor]) -> None:
        for i, cluster in enumerate(self.clusters["lec"]):
            cluster.feedback(targets[i])


# -----------------------------------------------------------------------------------
class VectorHaSH(nn.Module):
    def __init__(self, params: Optional[ModelParams] = None):
        super().__init__()
        self.params = params or ModelParams()
        self.mec_layerII = MECLayerII(scales=params.grid_sizes)
        self.lec_layerII = LECLayerII(contexts=params.contexts_size)
        self.dg = DG(params.latent_size, *params.ec_units)
        self.ca3 = CA3(params.latent_size, params.substate_size, *params.ec_units)

    # -----------------------------------------------------------------------------------
    def forward(self, position: Tuple[int, int], context: List[int]) -> Tensor:
        self.mec_targets = self.mec_layerII.encode(position)
        self.lec_targets = self.lec_layerII.encode(context)
        mec_flat = flatten(cat(self.mec_targets))
        lec_flat = flatten(cat(self.lec_targets))

        dg_pattern = self.dg(mec_flat, lec_flat)
        for _loop in range(self.params.state_loops):
            ca3_output = self.ca3(dg_pattern)  # Recurrent CA3 updates

        return ca3_output

    # -----------------------------------------------------------------------------------
    def feedback(self) -> None:
        self.ca3.feedback_mec(self.mec_targets)
        self.ca3.feedback_lec(self.lec_targets)


# -------------------------------------------------------------------------------------------
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
