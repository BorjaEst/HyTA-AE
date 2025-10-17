import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

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
from ehc_sn.loss import HomeostaticActivityLoss as SparsityLoss
from ehc_sn.metrics import MetricsLogger


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Model architecture parameters
    dg_dim: PositiveInt = Field(default=2000, gt=0, description="Dimensionality of the separation layer.")
    dg_sparsity: float = Field(default=0.05, ge=0.0, le=1.0, description="Sparsity target for pattern separator.")
    ca3_dim: PositiveInt = Field(default=400, gt=0, description="Number of units for latent representation.")
    ca1_dim: PositiveInt = Field(default=5000, gt=0, description="Number of units in the first hidden layer.")

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
class DFALayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_targets: int):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.target_features = n_targets
        self.register_buffer("activations", None)  # Starts without activation values
        self.register_buffer("fb_weight", torch.zeros(n_targets, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = torch.tanh(currents)
        return self.activations.detach()  # enforce locality

    def local_loss(self, error: Tensor) -> Tensor:
        delta = error.detach() @ self.fb_weight  # (batch_size, out_features)
        target = self.activations.detach() - delta  # same shape as activations
        return nn.functional.mse_loss(self.activations, target, reduction="mean")

    def reset_feedback(self) -> None:
        inv_limit = math.sqrt(self.target_features)
        self.fb_weight.bernoulli_(0.5).mul_(2).sub_(1).div_(inv_limit)


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_latent: int, n_h1: int, n_inputs: int):
        super().__init__()
        self.layer1 = DFALayer(n_inputs, n_h1, n_targets=n_inputs)
        self.layer2 = DFALayer(n_h1, n_latent, n_targets=n_inputs)

    def forward(self, x: Tensor) -> List[Tensor]:
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        return [h1, h2]

    def compute_loss(self, reconstruction: Tensor, targets: Tensor) -> Tensor:
        reconstruction_err = (reconstruction - targets).detach()
        loss_l1 = self.layer1.local_loss(reconstruction_err)
        loss_l2 = self.layer2.local_loss(reconstruction_err)
        return loss_l1 + loss_l2


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_latent: int, n_h1: int, n_inputs: int):
        super().__init__()
        self.layer2 = DFALayer(n_latent, n_h1, n_targets=n_inputs)
        self.layer1 = nn.Linear(n_h1, n_inputs)  # Last layer trained with BP

    def forward(self, latent: Tensor) -> List[Tensor]:
        h2 = self.layer2(latent)
        h1 = nn.functional.sigmoid(self.layer1(h2))
        return [h2, h1]

    def compute_loss(self, reconstruction: Tensor, targets: Tensor) -> Tensor:
        reconstruction_err = (reconstruction - targets).detach()
        loss_l2 = self.layer2.local_loss(reconstruction_err)
        loss_l1 = nn.functional.binary_cross_entropy(reconstruction, targets, reduction="none")
        return loss_l2 + loss_l1.sum(dim=1).mean()


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, dg_dim: int, dg_sparsity: float, ca3_dim: int, ca1_dim: int):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize encoder and decoder with DFA layers
        self.encoder = Encoder(ca3_dim, ca1_dim, n_inputs=25**2)
        self.separator = nn.Linear(ca3_dim, dg_dim)
        self.attractor = DFALayer(dg_dim, ca3_dim, n_targets=25**2)
        self.decoder = Decoder(ca3_dim, ca1_dim, n_inputs=25**2)

        # Input and output reshaping layers (25x25 maps)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 25))

        # Loss functions and metrics
        self.sparsity_loss = SparsityLoss(target_rate=dg_sparsity, min_active=int(math.log2(dg_dim)))
        self.metrics = MetricsLogger(self)

    # -----------------------------------------------------------------------------------
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

    def _expand(self, latent: Tensor) -> Tensor:
        return torch.relu(torch.tanh(self.separator(latent)))  # Enforce non-negativity

    def _compress(self, patterns: Tensor) -> Tensor:
        return self.attractor(patterns)  # Activation already applied in DFALayer

    @torch.inference_mode()
    def recall(self, latent: Tensor) -> Tensor:
        patterns = self._expand(latent)
        return self._compress(patterns)

    # -----------------------------------------------------------------------------------
    def signals(self, batch: Tuple[Tensor, Tensor]) -> List[Tensor]:
        _sensors, targets = batch
        hidden_pre, latent_pre = self.encoder(self.flatten(targets))
        patterns = self._expand(latent_pre)
        latent_post = self._compress(patterns)
        hidden_post, output = self.decoder(latent_post)
        reconstruction = self.unflatten(output)
        return [hidden_pre, latent_pre, patterns, latent_post, hidden_post, reconstruction]

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        _, _, patterns, state, _, reconstruction = self.signals(batch)
        return reconstruction, patterns, state

    # -----------------------------------------------------------------------------------
    def compute_loss(self, reconstruction: Tensor, patterns: Tensor, batch: Tuple[Tensor, Tensor]) -> Tensor:
        sensors, targets = batch
        mask = sensors[:, 1]  # 1 = visible, 0 = hidden

        # FCMT: Masked Error and BCE using weighted loss (only visible pixels contribute)
        # Flatten spatial dimensions
        recon_flat = reconstruction.flatten(start_dim=1)
        target_flat = targets.flatten(start_dim=1)
        mask_flat = mask.flatten(start_dim=1)

        # Compute masked values for DFA feedback
        recon_masked = recon_flat * mask_flat
        target_masked = target_flat * mask_flat

        # Compute losses per module and sum
        loss_encoder = self.encoder.compute_loss(recon_masked, target_masked)
        loss_separator = self.sparsity_loss(patterns)
        loss_attractor = self.attractor.local_loss(recon_masked - target_masked)
        loss_decoder = self.decoder.compute_loss(recon_masked, target_masked)

        return loss_encoder + loss_separator + loss_attractor + loss_decoder

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        output = self(batch)  # Forward pass
        reconstruction, patterns, state = output
        global_loss = self.compute_loss(reconstruction, patterns, batch)
        self.manual_backward(global_loss)
        self.optimizers().step()
        self.metrics.log_training(batch, output)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        all_signals = self.signals(batch)
        self.metrics.log_validation(batch, all_signals)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule) -> None:
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    _, targets = batch = next(iter(test_dataloader))
    with torch.inference_mode():
        reconstruction, patterns, _latent = model(batch)

    # Figure 1: Reconstruction map comparing inputs and outputs
    fig_reconstruction = ReconstructionMapFigure()
    _ = fig_reconstruction.plot(targets, reconstruction)
    plt.show()

    # Figure 2: Sparsity plot showing patters activations
    sparsity_figure = SparsityFigure()
    _ = sparsity_figure.plot(patterns)
    plt.show()

    # Probe decoder by activating one latent (CA3) unit at a time
    latents = torch.eye(model.hparams.ca3_dim)[:18]
    reconstructions = model.decode(latents)

    # Figure 3: Decoder montage showing individual latent unit reconstructions
    decoder_montage_figure = DecoderMontageFigure()
    _ = decoder_montage_figure.plot(latents, reconstructions)
    plt.show()


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- Running HyTA Full-Context Masked Training (FCMT) Experiment ---")

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
        dg_dim=experiment.dg_dim,
        dg_sparsity=experiment.dg_sparsity,
        ca3_dim=experiment.ca3_dim,
        ca1_dim=experiment.ca1_dim,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=experiment.max_epochs,
        callbacks=[ModelCheckpoint(every_n_epochs=experiment.checkpoint_freq, save_weights_only=True)],
        logger=TensorBoardLogger(experiment.log_dir, name=experiment.experiment_name),
        profiler=None,  # "simple" for basic profiling and "advanced" for detailed profiling
    )

    # Train till end of training or keyboard interup
    try:
        trainer.fit(model, datamodule=datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        gen_figures(model, datamodule)
