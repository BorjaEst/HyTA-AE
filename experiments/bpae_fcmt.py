import math
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, flatten, nn, unflatten
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
    latent_size: PositiveInt = Field(default=2000, gt=0, description="Dimensionality of the latent code.")
    layer2_size: PositiveInt = Field(default=400, gt=0, description="Number of hidden units in layer 2.")
    layer1_size: PositiveInt = Field(default=5000, gt=0, description="Number of hidden units in layer 1.")
    sparsity_lambda: float = Field(default=0.00, ge=0.0, le=1.0, description="Weight of the sparsity loss term.")

    # Data and augmentation parameters
    mask_ratio: float = Field(default=0.55, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")
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
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int):
        super().__init__()
        self.layer1 = nn.Linear(n_inputs, n_h1)
        self.layer2 = nn.Linear(n_h1, n_h2)

    def forward(self, x: Tensor) -> List[Tensor]:
        h1 = nn.functional.gelu(self.layer1(x))
        h2 = nn.functional.gelu(self.layer2(h1))
        return [h1, h2]


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_latents: int, n_h2: int, n_h1: int):
        super().__init__()
        self.layer2 = nn.Linear(n_latents, n_h2)
        self.layer1 = nn.Linear(n_h2, n_h1)

    def forward(self, latent: Tensor) -> List[Tensor]:
        h2 = nn.functional.gelu(self.layer2(latent))
        h1 = nn.functional.gelu(self.layer1(h2))
        return [h1, h2]


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, latent_size: int, layer2_size: int, layer1_size: int, sparsity_lambda: float):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize encoder and decoder
        self.encoder = Encoder(n_inputs=25**2, n_h1=layer1_size, n_h2=layer2_size)
        self.latent = nn.Linear(layer2_size, latent_size)
        self.decoder = Decoder(latent_size, n_h2=layer2_size, n_h1=layer1_size)
        self.output = nn.Linear(layer1_size, 25**2)

        # Loss functions and metrics
        self.reconstruction_loss = nn.BCELoss(reduction="none")  # Masked BCE loss
        self.sparsity_loss = SparsityLoss(center=True)
        self.metrics = MetricsLogger(self)

    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder.parameters(), "lr": 2e-5},
            {"params": self.latent.parameters(), "lr": 2e-5},
            {"params": self.decoder.parameters(), "lr": 1e-3},
            {"params": self.output.parameters(), "lr": 1e-3},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def _encode(self, inputs: Tensor) -> Tensor:
        encoder_signals = self.encoder(flatten(inputs, start_dim=1))
        latent = self.latent(encoder_signals[-1])
        return nn.functional.gelu(latent)  # Nonlinearity on latent code

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        """Encode sensors into latent space using the DFA-based encoder."""
        return self._encode(sensors)

    def _decode(self, latent: Tensor) -> Tensor:
        decoder_signals = self.decoder(latent)
        logits = unflatten(self.output(decoder_signals[0]), 1, (25, 25))
        return torch.sigmoid(logits)

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent codes into map space using the HTL-based decoder."""
        return self._decode(latent)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        _sensors, targets = batch
        latent = self._encode(targets)  # We use full context for prediction
        reconstruction = self._decode(latent)  # Remove .detach() to allow gradients
        return reconstruction, latent

    # -----------------------------------------------------------------------------------
    def compute_loss(self, output: Tuple[Tensor, Tensor], batch: Tuple[Tensor, Tensor]) -> Tensor:
        reconstruction, latent = output
        sensors, targets = batch
        mask = sensors[:, 1]  # 1 = visible, 0 = hidden

        # FCMT: Masked BCE using weighted loss (only visible pixels contribute)
        # Flatten spatial dimensions
        recon_flat = reconstruction.flatten(start_dim=1)
        target_flat = targets.flatten(start_dim=1)
        mask_flat = mask.flatten(start_dim=1)

        # Compute BCE loss only on visible pixels
        bce_per_pixel = self.reconstruction_loss(recon_flat, target_flat)
        loss_rec = (bce_per_pixel * mask_flat).sum(dim=1).mean()  # Average over batch

        # Sparsity loss on latent
        loss_sparse = self.sparsity_loss(latent)

        return loss_rec + self.hparams.sparsity_lambda * loss_sparse

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        output = reconstruction, latent = self(batch)
        global_loss = self.compute_loss(output, batch)
        self.manual_backward(global_loss)
        self.optimizers().step()

        # Log metrics using MetricsLogger
        _, targets = batch
        self.metrics.log_all_training(reconstruction, latent, targets, include_stats=False)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        _sensors, targets = batch
        reconstruction, latent = self(batch)

        # Compute validation loss (same masked BCE as training)
        output = (reconstruction, latent)
        global_loss = self.compute_loss(output, batch)

        encoder_signals = self.encoder(flatten(targets, start_dim=1))
        decoder_signals = self.decoder(latent)

        # Log validation metrics using MetricsLogger
        self.metrics.log_all_validation(reconstruction, latent, targets, include_stats=True)
        self.metrics.log_layer_alignment(decoder_signals[0], encoder_signals[0], layer_idx=1, prefix="val")
        self.metrics.log_layer_alignment(decoder_signals[1], encoder_signals[1], layer_idx=2, prefix="val")

        # HParams plugin: provide a single comparable metric
        self.log("hp_metric", global_loss, on_epoch=True, prog_bar=False)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule) -> None:
    """Generate reconstruction and sparsity figures from model outputs."""
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    _, targets = batch = next(iter(test_dataloader))
    with torch.inference_mode():
        outputs, activations = model(batch)

    # Figure 1: Reconstruction map comparing inputs and outputs
    fig_reconstruction = ReconstructionMapFigure()
    _ = fig_reconstruction.plot(targets, outputs)
    plt.show()

    # Figure 2: Sparsity plot showing latent activations
    sparsity_figure = SparsityFigure()
    _ = sparsity_figure.plot(activations)
    plt.show()

    # Probe decoder by activating one latent unit at a time (one-hot codes)
    latent_dim = activations.shape[1]  # Number of latent units
    latents = torch.eye(latent_dim)[:18]  # One-hot encoding for each unit

    # Generate decoder outputs for one-hot latents
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
        latent_size=experiment.latent_size,
        layer2_size=experiment.layer2_size,
        layer1_size=experiment.layer1_size,
        sparsity_lambda=experiment.sparsity_lambda,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=experiment.max_epochs,
        callbacks=[ModelCheckpoint(every_n_epochs=experiment.checkpoint_freq, save_weights_only=True)],
        logger=TensorBoardLogger(experiment.log_dir, name=experiment.experiment_name),
        profiler="simple",
    )

    try:  # Train till end of training or keyboard interup
        trainer.fit(model, datamodule=datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        gen_figures(model, datamodule)
