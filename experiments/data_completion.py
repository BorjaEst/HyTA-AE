import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, flatten, nn, unflatten
from torch.optim import Adam, Optimizer

from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
from ehc_sn.core import ann
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import BaseTrainer
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.decoder_montage import DecoderMontageParams as Fig3Params
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapParams as Fig1Params
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.figures.sparsity import SparsityParams as Fig2Params
from ehc_sn.modules import dfa, htl
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """Configuration settings for the hybrid autoencoder experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Encoder and decoder components
    latent_units: int = Field(default=2048, gt=0, description="Dimensionality of the latent code.")
    layer2_units: int = Field(default=512, gt=0, description="Number of hidden units per layer.")
    layer1_units: int = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[int] = Field([25, 25], description="Dimensionality of the input and output.")

    # Data and augmentation parameters
    augmentation: ComposeParams = Field(default_factory=ComposeParams, description="Data augmentation parameters")
    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")

    # Figures parameters
    figure_1: Fig1Params = Field(default_factory=Fig1Params, description="Reconstruction figure parameters")
    figure_2: Fig2Params = Field(default_factory=Fig2Params, description="Sparsity figure parameters")
    figure_3: Fig3Params = Field(default_factory=Fig3Params, description="Decoder montage figure parameters")

    # Training Settings
    max_epochs: int = Field(default=200, ge=1, le=1000, description="Maximum training epochs")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field(..., description="Experiment name")
    checkpoint_freq: int = Field(default=5, ge=1, le=50, description="Checkpoint frequency")

    @field_validator("output_shape")
    def check_output_shape(cls, v: List[int]) -> List[int]:
        if any(dim <= 0 for dim in v):
            raise ValueError("output_shape must be a list of positive integers; e.g. [H, W, C]")
        return v


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int):
        super().__init__()
        self.layer1 = ann.Layer(dfa.Linear(n_inputs, n_h1, error_features=n_inputs), nn.GELU())
        self.layer2 = ann.Layer(dfa.Linear(n_h1, n_h2, error_features=n_inputs), nn.GELU())

    def forward(self, sensors: Tensor) -> List[Tensor]:
        h1 = self.layer1(sensors)
        h2 = self.layer2(h1)
        return [h1, h2]

    def feedback(self, reconstruction_err: Tensor) -> None:
        self.layer2.synapses.feedback(reconstruction_err, context=self.layer2.neurons)
        self.layer1.synapses.feedback(reconstruction_err, context=self.layer1.neurons)


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer2 = ann.Layer(htl.Linear(n_latents, n_h2), nn.GELU())
        self.layer1 = ann.Layer(htl.Linear(n_h2, n_h1), nn.GELU())

    def forward(self, latent: Tensor) -> List[Tensor]:
        h2 = self.layer2(latent)
        h1 = self.layer1(h2)
        return [h1, h2]

    def feedback(self, encoder: Encoder, latent: Tensor) -> None:
        self.layer2.synapses.feedback(encoder.layer2.neurons, context=latent)
        self.layer1.synapses.feedback(encoder.layer1.neurons, context=encoder.layer2.neurons)


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, n_output: int, n_layer1: int, n_layer2: int, n_latent: int):
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.automatic_optimization = False

        # Initialize encoder and decoder with DFA layers
        self.encoder = Encoder(n_output, n_layer1, n_layer2)
        self.latent = ann.Layer(nn.Linear(n_layer2, n_latent), nn.GELU())
        self.decoder = Decoder(n_layer1, n_layer2, n_latent)
        self.output = ann.Layer(nn.Linear(n_layer1, n_output), nn.Sigmoid())

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(center=True)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder.parameters(), "lr": 1e-4},
            {"params": self.latent.parameters(), "lr": 1e-4},
            {"params": self.decoder.parameters(), "lr": 1e-3},
            {"params": self.output.parameters(), "lr": 1e-2},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        sensors, targets = batch
        encoder_signals = self.encoder(flatten(sensors, start_dim=1))
        latent = self.latent(encoder_signals[-1].detach())
        decoder_signals = self.decoder(latent)
        reconstruction = self.output(decoder_signals[0].detach())
        return unflatten(reconstruction, 1, sensors.shape[1:]), latent

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        encoder_signals = self.encoder(flatten(sensors, start_dim=1))
        return self.latent(encoder_signals[-1].detach())

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        decoder_signals = self.decoder(latent)
        reconstruction = self.output(decoder_signals[0].detach())
        return unflatten(reconstruction, 1, self.config.output_shape)

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
        self.decoder.feedback(self.encoder, latent.detach())
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


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- Running Data Completion with Feedback Experiment ---")

    # Initialize experiment configuration
    experiment = Experiment(experiment_name="datcom_feedback")
    augmentation = Augmentation(experiment.augmentation)
    data_gen = DataGenerator(experiment.data, augmentation)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)

    # Initialize model with specified architecture
    model = Autoencoder(
        n_output=math.prod(experiment.output_shape),
        n_layer1=experiment.layer1_units,
        n_layer2=experiment.layer2_units,
        n_latent=experiment.latent_units,
    )

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
