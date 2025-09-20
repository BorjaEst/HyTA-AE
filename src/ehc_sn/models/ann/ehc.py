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
    dg_units: int = Field(default=2024, gt=0, description="Dimensionality of the Dentate Gyrus units.")
    ca3_units: int = Field(default=512, gt=0, description="Number of Cornu Ammonis area 3 units.")
    ca1_units: int = Field(default=1024, gt=0, description="Number of Cornu Ammonis area 1 units.")
    # subiculum_units: int = Field(default=2048, gt=0, description="Number of Subiculum units.")
    output_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Training parameters
    mec_lr: float = Field(2e-6, description="Learning rate for the mec.")
    hpc_lr: float = Field(1e-4, description="Learning rate for the hpc.")

    @field_validator("output_shape")
    def check_output_shape(cls, v: List[int]) -> List[int]:
        if any(dim <= 0 for dim in v):
            raise ValueError("output_shape must be a list of positive integers; e.g. [H, W, C]")
        return v

    @property
    def mec_kwargs(self) -> Dict[str, int]:
        """Return dictionary of mec layer sizes."""
        return {
            "dim_V": math.prod(self.output_shape),
            "dim_III": self.ca1_units,
            "dim_II": self.ca3_units,
        }

    @property
    def hpc_kwargs(self) -> Dict[str, int]:
        """Return dictionary of hpc layer sizes."""
        return {
            "dim_subiculum": math.prod(self.output_shape),
            "dim_ca1": self.ca1_units,
            "dim_ca3": self.ca3_units,
            "dim_dg": self.dg_units,
        }


# -------------------------------------------------------------------------------------------
class MEC(nn.Module):
    def __init__(self, dim_V: int, dim_III: int, dim_II: int):
        super().__init__()
        self.layerIII = ann.Layer(dfa.Linear(dim_V, dim_III, error_features=dim_V), nn.GELU())
        self.layerII = ann.Layer(dfa.Linear(dim_V, dim_II, error_features=dim_V), nn.GELU())

    def forward(self, sensors: Tensor) -> None:
        self.layerIII(sensors)
        self.layerII(sensors)

    def feedback(self, reconstruction_err: Tensor) -> None:
        self.layerII.synapses.feedback(reconstruction_err, context=self.layerII.neurons)
        self.layerIII.synapses.feedback(reconstruction_err, context=self.layerIII.neurons)


# -------------------------------------------------------------------------------------------
class HPC(nn.Module):
    def __init__(self, dim_subiculum: int, dim_ca1: int, dim_ca3: int, dim_dg: int):
        super().__init__()
        self.sparsity_loss = SparsityLoss(center=True)
        self.dg = ann.Layer(nn.Linear(dim_ca3, dim_dg), nn.GELU())  # !! we are assuming dim_ca3==dim_II
        self.ca3 = ann.Layer(srtp.Linear(dim_dg, dim_ca3), nn.GELU())
        self.ca1 = ann.Layer(srtp.Linear(dim_ca3, dim_ca1), nn.GELU())
        self.subiculum = ann.Layer(nn.Linear(dim_ca1, dim_subiculum), nn.Sigmoid())

    def forward(self, mec: MEC) -> None:
        self.dg(mec.layerII.neurons.detach())
        self.ca3(self.dg.neurons)
        self.ca1(self.ca3.neurons)
        self.subiculum(self.ca1.neurons.detach())  # !! we need to add output for the backprop loss and fix detach

    def feedback(self, mec: MEC) -> None:
        self.sparsity_loss(self.dg.neurons).backward()
        self.ca3.synapses.feedback(mec.layerII.neurons, context=self.dg.neurons)
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
        self.mec = MEC(**params.mec_kwargs)
        self.hpc = HPC(**params.hpc_kwargs)

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optm_pe = {"params": self.mec.parameters(), "lr": self.config.mec_lr}
        optm_pd = {"params": self.hpc.parameters(), "lr": self.config.hpc_lr}
        return Adam([optm_pe, optm_pd])

    # -----------------------------------------------------------------------------------
    def forward(self, sensors: Tensor) -> Tuple[Tensor, Tensor]:
        mec, hpc = self.mec, self.hpc
        mec(flatten(sensors, start_dim=1))
        hpc(mec)
        reconstruction = unflatten(hpc.subiculum.neurons, 1, sensors.shape[1:])
        return reconstruction, hpc.dg.neurons

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
