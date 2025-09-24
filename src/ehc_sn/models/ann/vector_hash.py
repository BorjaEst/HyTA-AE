from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, Field
from torch import Tensor, cat, nn

from ehc_sn.modules import drtp
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss
from ehc_sn.utils import encoding_utils


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Dentate Gyrus parameters
    latent_size: int = Field(256, gt=0, description="Dimensionality of the DG latent space.")
    substate_size: int = Field(32, gt=0, description="Dimensionality of each CA3 subcluster.")
    state_loops: int = Field(5, gt=0, description="Number of CA3 recurrent state updates per step.")

    # Toroidal manifold shapes
    grid_sizes: List[int] = Field([2, 3, 6], description="List of grid scales.")
    context_sizes: List[int] = Field([30] * 4, description="List of context vector sizes.")

    @property
    def mec_shape(self) -> List[int]:
        return [s**2 for s in self.grid_sizes]

    @property
    def lec_shape(self) -> List[int]:
        return self.context_sizes


# -------------------------------------------------------------------------------------------
class MECLayerII(nn.Module):
    def __init__(self, scales: List[int]):
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.long))
        self.register_buffer("strides", encoding_utils.compute_strides(self.scales))
        self.period = int(torch.prod(self.scales).item())

    def encode(self, positions: Tensor) -> List[Tensor]:
        indices = encoding_utils.coordinates_to_indices(positions, self.strides, self.scales, self.period)
        return encoding_utils.indices_to_onehot(indices, self.scales)

    def decode(self, grids: List[torch.Tensor]) -> torch.Tensor:
        return encoding_utils.onehot_to_coordinates(grids, self.scales, self.strides)

    def __len__(self) -> int:
        return len(self.scales)


# -------------------------------------------------------------------------------------------
class LECLayerII(nn.Module):
    def __init__(self, contexts: List[int]):
        super().__init__()
        self.register_buffer("contexts", torch.tensor(contexts, dtype=torch.long))

    def encode(self, contexts: torch.Tensor) -> List[Tensor]:
        return encoding_utils.batch_to_onehot(contexts, self.contexts)

    def decode(self, contexts: List[Tensor]) -> torch.Tensor:
        """Decode contexts using shared one-hot decoding utilities."""
        return encoding_utils.onehot_to_indices(contexts)

    def __len__(self) -> int:
        return len(self.contexts)


# -------------------------------------------------------------------------------------------
class DG(nn.Module):
    def __init__(self, latent_size: int, mec_shape: List[int], lec_shape: List[int]):
        super().__init__()
        self.sparsity_loss = SparsityLoss(center=True)  # Sparsity loss for DG activations
        syn_mec = nn.Linear(sum(mec_shape), latent_size)
        syn_lec = nn.Linear(sum(lec_shape), latent_size)
        self.synapses = nn.ModuleDict({"mec": syn_mec, "lec": syn_lec})
        self.activation = nn.ReLU()
        self.state: Optional[Tensor] = None

    def forward(self, mec_inputs: Tensor, lec_inputs: Tensor) -> Tensor:
        x = self.synapses["mec"](mec_inputs) + self.synapses["lec"](lec_inputs)
        self.state = self.activation(x)
        return self.state.detach()

    def feedback(self) -> None:
        # Apply sparsity loss to the current state if available
        self.sparsity_loss(self.state).backward()


# -------------------------------------------------------------------------------------------
class CA3Cluster(nn.Module):
    def __init__(self, latent_units: int, cluster_size: int, target_size: int):
        super().__init__()
        self.input_syn = nn.Linear(latent_units, cluster_size)
        self.activation = nn.ReLU()
        self.target_syn = drtp.Linear(cluster_size, target_features=target_size)
        self.state: Optional[Tensor] = None

    def forward(self, dg_input: Tensor, recurrent_input: Tensor) -> Tensor:
        x = self.input_syn(dg_input) + recurrent_input
        self.state = self.activation(x)
        return self.target_syn(self.state)  # Detaches internally

    def feedback(self, target: Tensor) -> None:
        # Apply DRTP feedback to the target synapse using the provided target
        self.target_syn.feedback(target)


# -------------------------------------------------------------------------------------------
class CA3(nn.Module):
    def __init__(self, latent_size: int, cluster_size: int, mec_shape: List[int], lec_shape: List[int]):
        super().__init__()
        self.n_clusters = len(mec_shape) + len(lec_shape)
        self.cluster_size = cluster_size
        # Shared recurrent transformation
        self.recurrent_syn = nn.Linear(self.units, self.units)
        mec_clusters = [CA3Cluster(latent_size, cluster_size, x) for x in mec_shape]
        lec_clusters = [CA3Cluster(latent_size, cluster_size, x) for x in lec_shape]
        clusters_dict = {"mec": nn.ModuleList(mec_clusters), "lec": nn.ModuleList(lec_clusters)}
        self.clusters = nn.ModuleDict(clusters_dict)

    @property
    def units(self) -> int:
        return self.n_clusters * self.cluster_size

    @property
    def state(self) -> Tensor:
        states = [c.state for c in list(self.clusters["mec"]) + list(self.clusters["lec"])]
        B = states[0].shape[0] if states[0] is not None else 1
        device = states[0].device if states[0] is not None else torch.device("cpu")
        return encoding_utils.concat_states(states, B, self.cluster_size, device).detach()

    def reset_states(self) -> None:
        for c in list(self.clusters["mec"]) + list(self.clusters["lec"]):
            c.state = None

    def forward(self, dg_pattern: Tensor) -> Tensor:
        recurrent_input = self.recurrent_syn(self.state.detach())  # Shared recurrent transformation
        # Split recurrent input for each cluster
        cluster_inputs = torch.split(recurrent_input, self.cluster_size, dim=-1)
        outputs_mec = [m(dg_pattern, cluster_inputs[i]) for i, m in enumerate(self.clusters["mec"])]
        outputs_lec = [m(dg_pattern, cluster_inputs[i]) for i, m in enumerate(self.clusters["lec"], i)]
        return cat(outputs_mec + outputs_lec, dim=-1)

    def feedback(self, cluster_id: str, targets: List[Tensor]) -> None:
        for i, cluster in enumerate(list(self.clusters[cluster_id])):
            cluster.feedback(targets[i])


# -----------------------------------------------------------------------------------
class VectorHaSH(nn.Module):
    def __init__(self, params: Optional[ModelParams] = None):
        super().__init__()
        self.params = p = params or ModelParams()
        self.ca3 = CA3(p.latent_size, p.substate_size, p.mec_shape, p.lec_shape)
        self.dg = DG(p.latent_size, p.mec_shape, p.lec_shape)
        self.mec_layerII = MECLayerII(scales=p.grid_sizes)
        self.lec_layerII = LECLayerII(contexts=p.context_sizes)

    def forward(self, positions: Tensor, contexts: Tensor) -> Tensor:
        self.mec_targets = self.mec_layerII.encode(positions)
        self.lec_targets = self.lec_layerII.encode(contexts)

        # Flatten and concatenate EC targets for DG input
        mec_flat = torch.cat(self.mec_targets, dim=-1).detach()
        lec_flat = torch.cat(self.lec_targets, dim=-1).detach()

        # DG and CA3 dynamics
        dg_pattern = self.dg(mec_flat, lec_flat)
        self.ca3.reset_states()
        for _ in range(self.params.state_loops):
            y = self.ca3(dg_pattern)

        return y.detach()

    def feedback(self) -> None:
        self.dg.feedback()  # Apply sparsity loss feedback in DG
        self.ca3.feedback("mec", self.mec_targets)
        self.ca3.feedback("lec", self.lec_targets)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test hippocampal state generation with tensor-only API
    print("=== Testing VectorHaSH with tensor-only encoder API ===")
    from torch.optim import Adam

    torch.manual_seed(0)

    params = ModelParams(grid_sizes=[2, 3, 6], context_sizes=[4, 5, 3])
    model = VectorHaSH(params)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=1e-3)

    batch_positions = torch.tensor([(0, 0), (17, 5), (35, 35)], dtype=torch.long)
    batch_contexts = torch.tensor([[0, 2, 1], [3, 4, 0], [1, 1, 2]], dtype=torch.long)

    print("VectorHaSH demo")
    print(f"MEC scales={model.mec_layerII.scales.tolist()}, period={model.mec_layerII.period}")
    for s in model.mec_layerII.scales:
        print(f"  scale={s.item()} -> grid shape=({s.item()}, {s.item()})")

    print(f"\nLEC contexts={model.lec_layerII.contexts.tolist()}")
    for i, c in enumerate(model.lec_layerII.contexts):
        print(f"  context[{i}] size={c.item()}")

    # Forward on full batch using tensor-only API
    Y = model.forward(batch_positions, batch_contexts)

    # DRTP feedback + update
    optimizer.zero_grad()
    model.feedback()
    optimizer.step()

    # Decode batch for inspection using tensor-only API
    mec_decoded_t = model.mec_layerII.decode(model.mec_targets)  # (B, 2)
    lec_decoded_t = model.lec_layerII.decode(model.lec_targets)  # (B, K)

    print(f"\nBatched CA3 output shape across {batch_positions.shape[0]} samples: {tuple(Y.shape)}")
    for i in range(batch_positions.shape[0]):
        pos = tuple(batch_positions[i].tolist())
        ctx = batch_contexts[i].tolist()
        mec_dec = tuple(mec_decoded_t[i].tolist())
        lec_dec = lec_decoded_t[i].tolist()
        print(f"Sample {i+1}:")
        print(f"  Input position={pos}, context={ctx}")
        print(f"  Decoded MEC position={mec_dec}, decoded LEC context={lec_dec}")
