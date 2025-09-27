import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field, field_validator
from torch import Tensor, flatten, nn, unflatten
from torch.optim import Adam, Optimizer

from ehc_sn.core import ann
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.modules import dfa, htl
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -------------------------------------------------------------------------------------------
class ModelParams(BaseModel):
    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    # Encoder and decoder components
    latent_units: int = Field(default=32, gt=0, description="Dimensionality of the latent code.")
    layer2_units: int = Field(default=512, gt=0, description="Number of hidden units per layer.")
    layer1_units: int = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Training parameters
    encoder_lr: float = Field(2e-6, description="Learning rate for the encoder.")
    decoder_lr: float = Field(1e-4, description="Learning rate for the decoder.")

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
class Encoder(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer1 = ann.Layer(dfa.Linear(n_inputs, n_h1, error_features=n_inputs), nn.GELU())
        self.layer2 = ann.Layer(dfa.Linear(n_h1, n_h2, error_features=n_inputs), nn.GELU())
        self.latent = ann.Layer(nn.Linear(n_h2, n_latents), nn.GELU())

    def forward(self, sensors: Tensor) -> Tensor:
        x = self.layer1(sensors)
        x = self.layer2(x)
        return self.latent(x.detach())

    def feedback(self, reconstruction_err: Tensor) -> None:
        self.layer2.synapses.feedback(reconstruction_err, context=self.layer2.neurons)
        self.layer1.synapses.feedback(reconstruction_err, context=self.layer1.neurons)


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_outputs: int, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer2 = ann.Layer(htl.Linear(n_latents, n_h2), nn.GELU())
        self.layer1 = ann.Layer(htl.Linear(n_h2, n_h1), nn.GELU())
        self.output = ann.Layer(nn.Linear(n_h1, n_outputs), nn.Sigmoid())

    def forward(self, latent: Tensor) -> Tensor:
        x = self.layer2(latent)
        x = self.layer1(x)
        return self.output(x.detach())

    def feedback(self, encoder: Encoder) -> None:
        self.layer2.synapses.feedback(encoder.layer2.neurons, context=encoder.latent.neurons)
        self.layer1.synapses.feedback(encoder.layer1.neurons, context=encoder.layer2.neurons)


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.config = params
        self.automatic_optimization = False

        # Initialize encoder and decoder with DFA layers
        self.encoder = Encoder(*params.units())
        self.decoder = Decoder(*params.units())

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(center=True)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optm_pe = {"params": self.encoder.parameters(), "lr": self.config.encoder_lr}
        optm_pd = {"params": self.decoder.parameters(), "lr": self.config.decoder_lr}
        return Adam([optm_pe, optm_pd])

    # -----------------------------------------------------------------------------------
    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        sensors, targets = batch
        latent = self.encoder(flatten(sensors, start_dim=1))
        reconstruction = unflatten(self.decoder(latent), 1, sensors.shape[1:])
        return reconstruction, latent

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        return self.encoder(flatten(sensors, start_dim=1))

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        return unflatten(self.decoder(latent), 1, self.config.output_shape)

    # -----------------------------------------------------------------------------------
    def compute_feedback(self, outputs: Tensor, batch: Tensor) -> List[Tensor]:
        reconstruction, latent = outputs
        sensors, *_ = batch
        return [sensors, reconstruction, latent]

    # -----------------------------------------------------------------------------------
    def apply_feedback(self, feedback: List[Tensor]) -> None:
        (sensors, reconstruction, latent) = feedback
        reconstruction_err = flatten(reconstruction - sensors, start_dim=1)
        self.encoder.feedback(reconstruction_err)
        self.sparsity_loss(latent).backward()
        self.decoder.feedback(self.encoder)
        self.reconstruction_loss(reconstruction, sensors).backward()

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        outputs = self(batch)
        feedback = self.compute_feedback(outputs, batch)
        self.apply_feedback(feedback)
        self.optimizers().step()

    # -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tensor, batch_idx: int) -> List[Tensor]:
        outputs = self(batch)
        reconstruction_loss = nn.MSELoss(reduction="mean")(outputs[0], batch[0])
        sparsity_rate = (outputs[1] > 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, prog_bar=True)
        self.log("val/reconstruction_loss", reconstruction_loss, prog_bar=True)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Experiment to evaluate how well a hybrid autoencoder can reconstruct incomplete maps
    print("=== Data Completion Autoencoder Example ===")
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from matplotlib import pyplot as plt
    from pydantic_settings import BaseSettings, SettingsConfigDict

    # Import library utilities for experiment
    from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
    from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
    from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
    from ehc_sn.figures.decoder_montage import DecoderMontageFigure
    from ehc_sn.figures.decoder_montage import DecoderMontageParams as Fig3Params
    from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
    from ehc_sn.figures.reconstruction_map import ReconstructionMapParams as Fig1Params
    from ehc_sn.figures.sparsity import SparsityFigure
    from ehc_sn.figures.sparsity import SparsityParams as Fig2Params

    # -----------------------------------------------------------------------------------
    class Experiment(BaseSettings):
        """Configuration settings for the hybrid autoencoder experiment."""

        model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

        augmentation: ComposeParams = Field(default_factory=ComposeParams, description="Data augmentation parameters")
        data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
        datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
        model: ModelParams = Field(default_factory=ModelParams, description="Autoencoder parameters")

        figure_1: Fig1Params = Field(default_factory=Fig1Params, description="Reconstruction figure parameters")
        figure_2: Fig2Params = Field(default_factory=Fig2Params, description="Sparsity figure parameters")
        figure_3: Fig3Params = Field(default_factory=Fig3Params, description="Decoder montage figure parameters")

        # Training Settings
        max_epochs: int = Field(default=200, ge=1, le=1000, description="Maximum training epochs")

        # Logging and Output Settings
        log_dir: str = Field(default="logs", description="Directory for experiment logs")
        experiment_name: str = Field(..., description="Experiment name")
        checkpoint_freq: int = Field(default=5, ge=1, le=50, description="Checkpoint frequency")

    # -----------------------------------------------------------------------------------
    def gen_figures(model: Autoencoder, datamodule: BaseDataModule, experiment: Experiment) -> None:
        """Generate reconstruction and sparsity figures from model outputs."""
        model.eval()
        datamodule.setup("test")
        test_dataloader = datamodule.test_dataloader()

        _, targets = batch = next(iter(test_dataloader))
        with torch.inference_mode():
            outputs, activations = model(batch)

        # Figure 1: Reconstruction map comparing inputs and outputs
        fig_reconstruction = ReconstructionMapFigure(experiment.figure_1)
        _ = fig_reconstruction.plot(targets, outputs)
        plt.show()

        # Figure 2: Sparsity plot showing latent activations
        sparsity_figure = SparsityFigure(experiment.figure_2)
        _ = sparsity_figure.plot(activations)
        plt.show()

        latent_dim = experiment.model.latent_units  # Number of latent units
        latents = torch.eye(latent_dim)[:18]  # One-hot encoding for each unit

        # Generate decoder outputs for one-hot latents
        reconstructions = model.decode(latents)

        # Figure 3: Decoder montage showing individual latent unit reconstructions
        decoder_montage_figure = DecoderMontageFigure(experiment.figure_3)
        _ = decoder_montage_figure.plot(latents, reconstructions)
        plt.show()

    # -----------------------------------------------------------------------------------
    experiment = Experiment(experiment_name="datcom_feedback")
    augmentation = Augmentation(experiment.augmentation)
    data_gen = DataGenerator(experiment.data, augmentation)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)
    model = Autoencoder(experiment.model)

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=experiment.max_epochs,
        callbacks=[ModelCheckpoint(every_n_epochs=experiment.checkpoint_freq, save_weights_only=True)],
        logger=TensorBoardLogger(experiment.log_dir, name=experiment.experiment_name),
        profiler="simple",
    )

    # Train till end of training or keyboard interup
    try:
        trainer.fit(model, datamodule=datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        gen_figures(model, datamodule, experiment)
