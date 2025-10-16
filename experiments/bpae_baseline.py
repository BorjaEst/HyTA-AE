import math
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.loss import GramianOrthogonalityLoss as SparsityLoss
from ehc_sn.metrics import MetricsLogger


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Model architecture parameters
    separator_dim: PositiveInt = Field(default=2000, gt=0, description="Dimensionality of the separation layer.")
    latent_dim: PositiveInt = Field(default=400, gt=0, description="Number of units for latent representation.")
    hidden_dim: PositiveInt = Field(default=5000, gt=0, description="Number of units in the first hidden layer.")
    sparsity_lambda: float = Field(default=0.00, ge=0.0, le=1.0, description="Weight of the sparsity loss term.")

    # Data and augmentation parameters
    mask_ratio: float = Field(default=0.00, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")
    seed: int = Field(default=0, ge=0, description="Random seed for reproducibility")

    # Training Settings
    max_epochs: PositiveInt = Field(default=200, ge=1, le=1000, description="Maximum training epochs")
    batch_size: PositiveInt = Field(default=32, gt=0, le=1024, description="Batch size for training")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field(__file__.split("/")[-1].replace(".py", ""), description="Experiment name")
    checkpoint_freq: PositiveInt = Field(default=50, ge=1, le=50, description="Checkpoint frequency")


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_latent: int, n_h1: int, n_inputs: int):
        super().__init__()
        self.layer1 = nn.Linear(n_inputs, n_h1)
        self.layer2 = nn.Linear(n_h1, n_latent)

    def forward(self, x: Tensor) -> List[Tensor]:
        h1 = nn.functional.gelu(self.layer1(x))
        h2 = nn.functional.gelu(self.layer2(h1))
        return [h1, h2]


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_latent: int, n_h1: int, n_inputs: int):
        super().__init__()
        self.layer2 = nn.Linear(n_latent, n_h1)
        self.layer1 = nn.Linear(n_h1, n_inputs)

    def forward(self, latent: Tensor) -> List[Tensor]:
        h2 = nn.functional.gelu(self.layer2(latent))
        h1 = nn.functional.sigmoid(self.layer1(h2))
        return [h2, h1]


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, separator_dim: int, latent_dim: int, hidden_dim: int, sparsity_lambda: float):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize encoder and decoder
        self.encoder = Encoder(latent_dim, hidden_dim, n_inputs=25**2)
        self.separator = nn.Linear(latent_dim, separator_dim)  # Pattern separation layer
        self.attractor = nn.Linear(separator_dim, latent_dim)  # Future work: Dynamics
        self.decoder = Decoder(latent_dim, hidden_dim, n_inputs=25**2)

        # Input and output reshaping layers (25x25 maps)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 25))

        # Loss functions and metrics
        self.reconstruction_loss = nn.BCELoss(reduction="none")  # Masked BCE loss
        self.sparsity_loss = SparsityLoss(center=True)
        self.metrics = MetricsLogger(self, eps_sparsity=0.02)

    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder.parameters(), "lr": 2e-5},
            {"params": self.separator.parameters(), "lr": 2e-5},
            {"params": self.attractor.parameters(), "lr": 1e-3},
            {"params": self.decoder.parameters(), "lr": 1e-3},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def _encode(self, inputs: Tensor) -> Tensor:
        inputs = self.flatten(inputs)  # Flatten 25x25 maps to vectors
        return self.encoder(inputs)[-1]

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        return self._encode(sensors)

    def _decode(self, latent: Tensor) -> Tensor:
        outputs = self.decoder(latent)[-1]
        return self.unflatten(outputs)  # Reshape vectors back to 25x25

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        return self._decode(latent)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        _sensors, targets = batch
        latent_pre = self._encode(targets)  # We use full context for inference
        patterns = nn.functional.gelu(self.separator(latent_pre))
        state = latent_post = nn.functional.gelu(self.attractor(patterns))
        reconstruction = self._decode(latent_post)
        return reconstruction, patterns, state

    def signals(self, batch: Tuple[Tensor, Tensor]) -> List[Tensor]:
        """
        Compute all intermediate activations for metrics logging.

        Returns: [h1_enc, h2_enc, patterns, latent_post, h2_dec, reconstruction]
        """
        _sensors, targets = batch
        encoder_signals = self.encoder(self.flatten(targets))  # [h1, h2]
        patterns = nn.functional.gelu(self.separator(encoder_signals[-1]))
        latent_post = nn.functional.gelu(self.attractor(patterns))
        decoder_signals = self.decoder(latent_post)  # [reconstruction_flat, h2_dec]
        reconstruction = self.unflatten(decoder_signals[-1])

        # Return: [h1_enc, h2_enc, patterns, latent_post, h2_dec, reconstruction]
        return [encoder_signals[0], encoder_signals[1], patterns, latent_post, decoder_signals[1], reconstruction]

    # -----------------------------------------------------------------------------------
    def compute_loss(self, reconstruction: Tensor, patterns: Tensor, batch: Tuple[Tensor, Tensor]) -> Tensor:
        sensors, targets = batch
        mask = sensors[:, 1:2]  # Shape (batch, 1, H, W); 1 = visible, 0 = hidden

        # FCMT: Masked BCE using weighted loss (only visible pixels contribute)
        recon_flat = reconstruction.flatten(start_dim=1)
        target_flat = targets.flatten(start_dim=1)
        mask_flat = mask.flatten(start_dim=1)

        # Compute BCE loss only on visible pixels
        bce_per_pixel = self.reconstruction_loss(recon_flat, target_flat)
        loss_rec = (bce_per_pixel * mask_flat).sum(dim=1).mean()  # Average over batch

        # Sparsity loss on pattern separation layer
        loss_sparse = self.sparsity_loss(patterns)

        return loss_rec + self.hparams.sparsity_lambda * loss_sparse

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        output = self(batch)  # Forward pass to get (reconstruction, patterns, state)
        reconstruction, patterns, state = output
        global_loss = self.compute_loss(reconstruction, patterns, batch)
        self.manual_backward(global_loss)
        self.optimizers().step()
        self.metrics.log_training(batch, output)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        all_signals = self.signals(batch)  # [h1_enc, h2_enc, patterns, latent_post, h2_dec, reconstruction]
        self.metrics.log_validation(batch, all_signals)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule) -> None:
    """Generate reconstruction and sparsity figures from model outputs."""
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    batch = next(iter(test_dataloader))
    _, targets = batch
    with torch.inference_mode():
        reconstruction, patterns, latent_post = model(batch)

    # Figure 1: Reconstruction map comparing inputs and outputs
    fig_reconstruction = ReconstructionMapFigure()
    _ = fig_reconstruction.plot(targets, reconstruction)
    plt.show()

    # Figure 2: Sparsity plot showing patterns activations
    sparsity_figure = SparsityFigure()
    _ = sparsity_figure.plot(patterns)
    plt.show()

    # Probe decoder by activating one latent (CA3) unit at a time
    latents = torch.eye(model.hparams.latent_dim)[:18]
    reconstructions = model.decode(latents)

    # Figure 3: Decoder montage showing individual latent unit reconstructions
    decoder_montage_figure = DecoderMontageFigure()
    _ = decoder_montage_figure.plot(latents, reconstructions)
    plt.show()


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- Running Baseline Sparse BP Experiment ---")

    # Initialize experiment configuration
    experiment = Experiment()
    composition_params = ComposeParams(mask_ratio=experiment.mask_ratio)
    data_params = DataParams(seed=experiment.seed, invert_walls=False)
    datamodule_params = DataModuleParams(batch_size=experiment.batch_size, num_samples=4000)

    # Create data generator and datamodule
    augmentation = Augmentation(composition_params)
    data_gen = DataGenerator(data_params, augmentation)
    datamodule = BaseDataModule(data_gen, datamodule_params)

    # Initialize model with specified architecture
    model = Autoencoder(
        separator_dim=experiment.separator_dim,
        latent_dim=experiment.latent_dim,
        hidden_dim=experiment.hidden_dim,
        sparsity_lambda=experiment.sparsity_lambda,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=experiment.max_epochs,
        callbacks=[ModelCheckpoint(every_n_epochs=experiment.checkpoint_freq, save_weights_only=True)],
        logger=TensorBoardLogger(experiment.log_dir, name=experiment.experiment_name),
        profiler=None,  # "simple" for basic profiling and "advanced" for detailed profiling
    )

    try:  # Train till end of training or keyboard interup
        trainer.fit(model, datamodule=datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        gen_figures(model, datamodule)
