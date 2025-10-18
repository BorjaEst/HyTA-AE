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
from ehc_sn.loss import HoyerActivityLoss as SparsityLoss
from ehc_sn.metrics import MetricsLogger


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Model architecture parameters
    separator_dim: PositiveInt = Field(default=2000, gt=0, description="Dimensionality of the separation layer.")
    sparsity_lambda: float = Field(default=0.00, ge=0.0, le=1.0, description="Weight of the sparsity loss term.")
    latent_dim: PositiveInt = Field(default=400, gt=0, description="Number of units for latent representation.")
    hidden_dim: PositiveInt = Field(default=5000, gt=0, description="Number of units in the first hidden layer.")

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
class BPLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = nn.functional.gelu(currents)
        return self.activations


# -------------------------------------------------------------------------------------------
class DGLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, sparsity_lambda: float):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values
        self.register_buffer("currents", None)  # Starts without current values
        self.sparsity_loss = SparsityLoss()  # Hoyer sparsity loss
        self.sparsity_lambda = sparsity_lambda

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        self.currents = super().forward(*args, **kwargs)
        self.activations = nn.functional.gelu(self.currents)
        return self.activations

    def local_loss(self) -> Tensor:
        activations = nn.functional.gelu(self.currents.detach())
        loss = self.sparsity_loss(activations)  # No sparsity on BP path
        return self.sparsity_lambda * loss


# -------------------------------------------------------------------------------------------
class OUTLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values
        self.reconstruction_loss = nn.BCELoss(reduction="none")

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = torch.sigmoid(currents)
        return self.activations

    def local_loss(self, target: Tensor) -> Tensor:
        return self.reconstruction_loss(self.activations, target.detach())


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, separator_dim: int, sparsity_lambda: float, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.metrics = MetricsLogger(self, eps_sparsity=0.02)

        # Initialize encoder and decoder
        self.encoder_l1 = BPLayer(25**2, hidden_dim)
        self.encoder_l2 = BPLayer(hidden_dim, latent_dim)
        self.separator = DGLayer(latent_dim, separator_dim, sparsity_lambda)
        self.attractor = BPLayer(separator_dim, latent_dim)
        self.decoder_l1 = BPLayer(latent_dim, hidden_dim)
        self.output = OUTLayer(hidden_dim, 25**2)

        # Input and output reshaping layers (25x25 maps)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 25))

    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder_l1.parameters(), "lr": 2e-5},
            {"params": self.encoder_l2.parameters(), "lr": 2e-5},
            {"params": self.separator.parameters(), "lr": 2e-5},
            {"params": self.attractor.parameters(), "lr": 1e-3},
            {"params": self.decoder_l1.parameters(), "lr": 1e-3},
            {"params": self.output.parameters(), "lr": 1e-3},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def encode(self, inputs: Tensor) -> Tensor:
        inputs = self.flatten(inputs)
        hidden = self.encoder_l1(inputs)
        return self.encoder_l2(hidden)

    def decode(self, latent: Tensor) -> Tensor:
        hidden = self.decoder_l1(latent)
        output = self.output(hidden)
        return self.unflatten(output)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        sensors, _targets = batch
        latent = self.encode(sensors[:, 0])  # Use only the input channel
        return self.decode(latent)

    # -----------------------------------------------------------------------------------
    def signals(self, batch: Tuple[Tensor, Tensor]) -> List[Tensor]:
        (_sensors, targets), signals = batch, []
        inputs = self.flatten(targets)

        signals.append(self.encoder_l1(inputs))  # hidden_pre
        signals.append(self.encoder_l2(signals[-1]))  # latent_pre
        signals.append(self.separator(signals[-1]))  # patterns
        signals.append(self.attractor(signals[-1]))  # latent_post
        signals.append(self.decoder_l1(signals[-1]))  # hidden_post
        output = self.output(signals[-1])

        reconstruction = self.unflatten(output)
        signals.append(reconstruction)  # reconstruction

        return signals

    # -----------------------------------------------------------------------------------
    def compute_loss(self, signals: List[Tensor], batch: Tuple[Tensor, Tensor]) -> Tensor:
        _, _, _, _, _, reconstruction = signals
        sensors, _targets = batch

        # FCMT: Masked BCE using weighted loss (only visible pixels contribute)
        # Flatten spatial dimensions
        recon_flat = reconstruction.flatten(start_dim=1)
        sensors_flat = sensors[:, 0].flatten(start_dim=1)
        mask_flat = sensors[:, 1].flatten(start_dim=1)
        completed_flat = sensors_flat + (1 - mask_flat) * recon_flat  # Fill in missing

        # Compute local losses for each module (they backprop)
        losses = [
            self.separator.local_loss(),  # This does not backprop to previous layers
            self.output.local_loss(completed_flat).sum(dim=1).mean(),  # BP
        ]

        return sum(losses)  # Total global loss

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        signals = _, _, patterns, latent, _, reconstruction = self.signals(batch)
        global_loss = self.compute_loss(signals, batch)
        self.manual_backward(global_loss)
        self.optimizers().step()
        self.metrics.log_training(batch, [reconstruction, patterns, latent])

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
        _, _, patterns, _, _, reconstruction = model.signals(batch)

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
        sparsity_lambda=experiment.sparsity_lambda,
        latent_dim=experiment.latent_dim,
        hidden_dim=experiment.hidden_dim,
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
