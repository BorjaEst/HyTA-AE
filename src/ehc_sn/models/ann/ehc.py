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

    # MEC and hpc components
    latent_units: int = Field(default=2024, gt=0, description="Dimensionality of the latent code.")
    layer2_units: int = Field(default=512, gt=0, description="Number of hidden units per layer.")
    layer1_units: int = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Training parameters
    mec_lr: float = Field(2e-6, description="Learning rate for the mec.")
    hpc_lr: float = Field(1e-4, description="Learning rate for the hpc.")

    def units(self) -> List[int]:
        """Return list of layer sizes from input to latent."""
        output_units = math.prod(self.output_shape)
        return [output_units, self.layer1_units, self.layer2_units, self.latent_units]

    @field_validator("output_shape")
    def check_output_shape(cls, v: List[int]) -> List[int]:
        if any(dim <= 0 for dim in v):
            raise ValueError("output_shape must be a list of positive integers; e.g. [H, W, C]")
        return v


# -------------------------------------------------------------------------------------------
class MEC(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layerIII = ann.Layer(dfa.Linear(n_inputs, n_h1, error_features=n_inputs), nn.GELU())
        self.layerII = ann.Layer(dfa.Linear(n_h1, n_h2, error_features=n_inputs), nn.GELU())
        self.dg = ann.Layer(dfa.Linear(n_h2, n_latents, error_features=n_inputs), nn.GELU())

    def forward(self, sensors: Tensor) -> Tensor:
        x = self.layerIII(sensors)
        x = self.layerII(x)
        return self.dg(x)

    def feedback(self, reconstruction_err: Tensor) -> None:
        self.layerII.synapses.feedback(reconstruction_err, context=self.layerII.neurons)
        self.layerIII.synapses.feedback(reconstruction_err, context=self.layerIII.neurons)


# -------------------------------------------------------------------------------------------
class HPC(nn.Module):
    def __init__(self, n_outputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.ca3 = ann.Layer(srtp.Linear(n_latents, n_h2), nn.GELU())
        self.ca1 = ann.Layer(srtp.Linear(n_h2, n_h1), nn.GELU())
        self.subiculum = ann.Layer(nn.Linear(n_h1, n_outputs), nn.Sigmoid())

    def forward(self, latent: Tensor) -> Tensor:
        x = self.ca3(latent)
        x = self.ca1(x)
        return self.subiculum(x.detach())

    def feedback(self, mec: MEC) -> None:
        self.ca3.synapses.feedback(mec.layerII.neurons, context=mec.dg.neurons)
        self.ca1.synapses.feedback(mec.layerIII.neurons, context=mec.layerII.neurons)


# -------------------------------------------------------------------------------------------
class EHC(pl.LightningModule):
    def __init__(self, params: ModelParams, trainer: BaseTrainer) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.config = params
        self.automatic_optimization = False
        self.trainer_module = trainer

        # Initialize mec and hpc with DFA layers
        self.mec = MEC(*params.units())
        self.hpc = HPC(*params.units())

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(center=True)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optm_pe = {"params": self.mec.parameters(), "lr": self.config.mec_lr}
        optm_pd = {"params": self.hpc.parameters(), "lr": self.config.hpc_lr}
        return Adam([optm_pe, optm_pd])

    # -----------------------------------------------------------------------------------
    def forward(self, sensors: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.mec(flatten(sensors, start_dim=1))
        reconstruction = unflatten(self.hpc(latent), 1, sensors.shape[1:])
        return reconstruction, latent

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        return self.mec(flatten(sensors, start_dim=1))

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        return unflatten(self.hpc(latent), 1, self.config.output_shape)

    # -----------------------------------------------------------------------------------
    def compute_feedback(self, outputs: Tensor, batch: Tensor) -> List[Tensor]:
        reconstruction, latent = outputs
        sensors, *_ = batch
        return [sensors, reconstruction, latent]

    # -----------------------------------------------------------------------------------
    def apply_feedback(self, feedback: List[Tensor]) -> None:
        (sensors, reconstruction, latent) = feedback
        reconstruction_err = flatten(reconstruction - sensors, start_dim=1)
        self.mec.feedback(reconstruction_err)
        self.sparsity_loss(latent).backward()
        self.hpc.feedback(self.mec)
        self.reconstruction_loss(reconstruction, sensors).backward()

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
