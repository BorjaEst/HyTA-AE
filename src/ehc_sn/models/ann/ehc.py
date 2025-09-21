import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, flatten, nn, unflatten
from torch.optim import Adam, Optimizer

from ehc_sn.core import ann
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.modules import dfa, srtp
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # MEC and HPC layer sizes
    n_layerIII: int = Field(default=1024, gt=0, description="Number of MEC layer III units.")
    n_layerII: int = Field(default=512, gt=0, description="Number of MEC layer II units.")
    n_dg: int = Field(default=2024, gt=0, description="Dimensionality of the Dentate Gyrus units.")
    n_ca3: int = Field(default=512, gt=0, description="Number of Cornu Ammonis area 3 units.")
    n_ca1: int = Field(default=1024, gt=0, description="Number of Cornu Ammonis area 1 units.")
    n_subiculum: int = Field(default=1024, gt=0, description="Number of Subiculum units.")
    maps_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Training parameters
    lr_layerII: float = Field(2e-6, description="Learning rate for the layer II parameters.")
    lr_layerIII: float = Field(2e-6, description="Learning rate for the layer III parameters.")
    lr_dg: float = Field(2e-6, description="Learning rate for the Dentate Gyrus parameters.")
    lr_ca3: float = Field(1e-4, description="Learning rate for the Cornu Ammonis area 3 parameters.")
    lr_ca1: float = Field(1e-4, description="Learning rate for the Cornu Ammonis area 1 parameters.")
    lr_subiculum: float = Field(1e-4, description="Learning rate for the subiculum parameters.")
    lr_outputs: float = Field(1e-4, description="Learning rate for the output layer parameters.")

    @field_validator("maps_shape")
    def check_output_shape(cls, v: List[int]) -> List[int]:
        if any(dim <= 0 for dim in v):
            raise ValueError("Must be a list of positive integers; e.g. [H, W]")
        return v

    @property
    def n_sensors(self) -> int:
        """Return number of input units."""
        return math.prod(self.maps_shape)


# # -------------------------------------------------------------------------------------------
class InputGateway(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensors: Tensor | None = None

    def forward(self, sensors: Tensor) -> Tensor:
        self.sensors = sensors.detach()  # detach to avoid gradients
        return flatten(self.sensors, start_dim=1)

    def dfa_error(self, reconstruction: Tensor):
        # Detach reconstruction to prevent building a graph used by MEC feedback
        return flatten(reconstruction.detach() - self.sensors, start_dim=1)


# -------------------------------------------------------------------------------------------
class MEC(nn.Module):
    def __init__(self, n_sensors: int, n_layerIII: int, n_layerII: int):
        super().__init__()
        self.layerIII = ann.Layer(dfa.Linear(n_sensors, n_layerIII, error_features=n_sensors), nn.GELU())
        self.layerII = ann.Layer(dfa.Linear(n_sensors, n_layerII, error_features=n_sensors), nn.GELU())

    def forward(self, sensors: Tensor) -> None:
        self.layerIII(sensors)
        self.layerII(sensors)

    def feedback(self, error: Tensor) -> None:
        self.layerII.synapses.feedback(error, context=self.layerII.neurons)
        self.layerIII.synapses.feedback(error, context=self.layerIII.neurons)


# -------------------------------------------------------------------------------------------
class HPC(nn.Module):
    def __init__(self, n_layerII: int, n_dg: int, n_ca3: int, n_ca1: int, n_subiculum: int):
        super().__init__()
        self.sparsity_loss = SparsityLoss(center=True)
        self.dg = ann.Layer(nn.Linear(n_layerII, n_dg), nn.ReLU())
        self.ca3 = ann.Layer(srtp.Linear(n_dg, n_ca3), nn.GELU())
        self.ca1 = ann.Layer(srtp.Linear(n_ca3, n_ca1), nn.GELU())
        self.subiculum = ann.Layer(nn.Linear(n_ca1, n_subiculum), nn.GELU())

    def forward(self, mec: MEC) -> None:
        # Detach MEC->DG to avoid sharing autograd graph with MEC feedback
        latent = self.dg(mec.layerII.neurons.detach())
        state = self.ca3(latent)
        graph_map = self.ca1(state)
        self.subiculum(graph_map.detach())  # !! We need to study subiculum deeper

    def feedback(self, mec: MEC) -> None:
        self.sparsity_loss(self.dg.neurons).backward()  # sparsity on DG
        self.ca3.synapses.feedback(mec.layerII.neurons, context=self.dg.neurons)
        self.ca1.synapses.feedback(mec.layerIII.neurons, context=mec.layerII.neurons)


class OutputHead(nn.Module):
    def __init__(self, maps_shape: List[int], n_ca1: int, n_subiculum: int):
        super().__init__()
        synapses_ca1 = nn.Linear(n_ca1, math.prod(maps_shape))
        synapses_subiculum = nn.Linear(n_subiculum, math.prod(maps_shape))
        self.synapses = nn.ModuleDict({"ca1": synapses_ca1, "subiculum": synapses_subiculum})
        self.output_shape = maps_shape
        self.loss = nn.BCELoss(reduction="mean")
        self.activations: Tensor | None = None

    def forward(self, hpc: HPC) -> Tensor:
        x1 = self.synapses["ca1"](hpc.ca1.neurons.detach())
        x2 = self.synapses["subiculum"](hpc.subiculum.neurons.detach())
        self.activations = torch.sigmoid(x1 + x2)
        return unflatten(self.activations, dim=1, sizes=self.output_shape)

    def feedback(self, sensors: Tensor) -> None:
        # Flatten targets to match internal flat activations (B, H*W)
        target = flatten(sensors.detach(), start_dim=1)
        self.loss(self.activations, target).backward()


# -------------------------------------------------------------------------------------------
class EHC(pl.LightningModule):
    def __init__(self, params: ModelParams, trainer: BaseTrainer) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.config = params
        self.automatic_optimization = False
        self.trainer_module = trainer

        # Initialize Gateway, MEC, HPC and Output layers
        self.inputs = InputGateway()
        self.mec = MEC(params.n_sensors, params.n_layerIII, params.n_layerII)
        self.hpc = HPC(params.n_layerII, params.n_dg, params.n_ca3, params.n_ca1, params.n_subiculum)
        self.output = OutputHead(params.maps_shape, params.n_ca1, params.n_subiculum)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optm_params = [
            {"params": self.mec.layerII.parameters(), "lr": self.config.lr_layerII},
            {"params": self.mec.layerIII.parameters(), "lr": self.config.lr_layerIII},
            {"params": self.hpc.dg.parameters(), "lr": self.config.lr_dg},
            {"params": self.hpc.ca3.parameters(), "lr": self.config.lr_ca3},
            {"params": self.hpc.ca1.parameters(), "lr": self.config.lr_ca1},
            {"params": self.hpc.subiculum.parameters(), "lr": self.config.lr_subiculum},
            {"params": self.output.parameters(), "lr": self.config.lr_outputs},
        ]
        return Adam(optm_params)

    # -----------------------------------------------------------------------------------
    def forward(self, sensors: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = self.inputs(sensors)
        self.mec(inputs)
        self.hpc(self.mec)
        return self.output(self.hpc), self.hpc.dg.neurons

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        inputs = self.inputs(sensors)
        self.mec(inputs)
        self.hpc.dg(self.mec.layerII.neurons.detach())
        return self.hpc.dg.neurons

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        self.hpc.ca3(latent)
        self.hpc.ca1(self.hpc.ca3.neurons)
        self.hpc.subiculum(self.hpc.ca1.neurons)
        return self.output(self.hpc)

    # -----------------------------------------------------------------------------------
    def compute_feedback(self, outputs: Tensor, batch: Tensor) -> List[Tensor]:
        reconstruction, *_ = outputs
        sensors, *_ = batch
        return [sensors, reconstruction]

    # -----------------------------------------------------------------------------------
    def apply_feedback(self, feedback: List[Tensor]) -> None:
        (sensors, reconstruction) = feedback
        reconstruction_err = self.inputs.dfa_error(reconstruction)
        self.mec.feedback(reconstruction_err)
        self.hpc.feedback(self.mec)
        self.output.feedback(sensors)

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.trainer_module.training_step(self, batch, batch_idx)

    # -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tensor, batch_idx: int) -> List[Tensor]:
        outputs = self(batch[0])
        reconstruction_loss = nn.MSELoss(reduction="mean")(outputs[0], batch[0])
        sparsity_rate = (outputs[1] > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, prog_bar=True)
        self.log("val/reconstruction_loss", reconstruction_loss, prog_bar=True)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage example
    pass  # TODO
