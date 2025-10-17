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
class DGLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, target_sparsity: float):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values
        self.sparsity_loss = SparsityLoss(target_sparsity, min_active=int(math.log2(n_out)))

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = torch.tanh(torch.relu(currents))
        return self.activations.detach()  # enforce locality

    def local_loss(self) -> Tensor:
        return self.sparsity_loss(self.activations)


# -------------------------------------------------------------------------------------------
class HTALayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = torch.tanh(currents)
        return self.activations.detach()  # enforce locality

    def local_loss(self, target: Tensor) -> Tensor:
        return nn.functional.mse_loss(self.activations, target, reduction="mean")


# -------------------------------------------------------------------------------------------
class OUTLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values
        self.reconstruction_loss = nn.BCELoss(reduction="none")

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = torch.sigmoid(currents)
        return self.activations.detach()  # enforce locality

    def local_loss(self, target: Tensor) -> Tensor:
        return self.reconstruction_loss(self.activations, target).sum(dim=1).mean()


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, dg_dim: int, dg_sparsity: float, ca3_dim: int, ca1_dim: int):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.metrics = MetricsLogger(self)

        # Initialize encoder and decoder with DFA layers
        self.encoder_l1 = DFALayer(n_in=25**2, n_out=ca1_dim, n_targets=25**2)
        self.encoder_l2 = DFALayer(n_in=ca1_dim, n_out=ca3_dim, n_targets=25**2)
        self.dg = DGLayer(n_in=ca3_dim, n_out=dg_dim, target_sparsity=dg_sparsity)
        self.ca3 = HTALayer(n_in=dg_dim, n_out=ca3_dim)
        self.ca1 = HTALayer(n_in=ca3_dim, n_out=ca1_dim)
        self.output = OUTLayer(n_in=ca1_dim, n_out=25**2)

        # Input and output reshaping layers (25x25 maps)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 25))

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder_l1.parameters(), "lr": 2e-5},
            {"params": self.encoder_l2.parameters(), "lr": 2e-5},
            {"params": self.dg.parameters(), "lr": 2e-5},
            {"params": self.ca3.parameters(), "lr": 1e-3},
            {"params": self.ca1.parameters(), "lr": 1e-3},
            {"params": self.output.parameters(), "lr": 1e-3},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def encode(self, inputs: Tensor) -> Tensor:
        inputs = self.flatten(inputs)
        hidden = self.encoder_l1(inputs)
        return self.encoder_l2(hidden)

    def decode(self, latent: Tensor) -> Tensor:
        hidden = self.ca1(latent)
        outputs = self.output(hidden)
        return self.unflatten(outputs)

    # -----------------------------------------------------------------------------------
    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        sensors, _targets = batch
        latent = self.encode(sensors[:, 0])  # Use only the input channel
        return self.decode(latent), latent

    def signals(self, batch: Tuple[Tensor, Tensor]) -> List[Tensor]:
        _sensors, targets = batch
        inputs = self.flatten(targets)
        hidden_pre = self.encoder_l1(inputs)
        latent_pre = self.encoder_l2(hidden_pre)
        patterns = self.dg(latent_pre)
        latent_post = self.ca3(patterns)
        hidden_post = self.ca1(latent_post)
        output = self.output(hidden_post)
        reconstruction = self.unflatten(output)
        return [hidden_pre, latent_pre, patterns, latent_post, hidden_post, reconstruction]

    # -----------------------------------------------------------------------------------
    def compute_loss(self, signals: List[Tensor], batch: Tuple[Tensor, Tensor]) -> Tensor:
        hidden_pre, latent_pre, _, _, _, reconstruction = signals
        sensors, targets = batch
        mask = sensors[:, 1]  # 1 = visible, 0 = hidden

        # FCMT: Masked Error and BCE using weighted loss (only visible pixels contribute)
        # Flatten spatial dimensions
        recon_flat = reconstruction.flatten(start_dim=1)
        target_flat = targets.flatten(start_dim=1)
        mask_flat = mask.flatten(start_dim=1)
        completed_flat = recon_flat * (1 - mask_flat) + target_flat * mask_flat

        # Compute masked values for DFA feedback
        recon_masked = recon_flat * mask_flat
        target_masked = target_flat * mask_flat

        # Compute local losses for each module
        losses = [
            self.encoder_l1.local_loss(recon_masked - target_masked),
            self.encoder_l2.local_loss(recon_masked - target_masked),
            self.dg.local_loss(),
            self.ca3.local_loss(latent_pre),
            self.ca1.local_loss(hidden_pre),
            self.output.local_loss(completed_flat),
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
        all_signals = self.signals(batch)
        self.metrics.log_validation(batch, all_signals)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule) -> None:
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    _, targets = batch = next(iter(test_dataloader))
    with torch.inference_mode():
        _, _, patterns, _, _, reconstruction = model.signals(batch)

    # Figure 1: Reconstruction map comparing inputs and outputs
    fig_reconstruction = ReconstructionMapFigure()
    _ = fig_reconstruction.plot(targets, reconstruction)
    plt.show()

    # Figure 2: Sparsity plot showing patters activations
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
    print(f"\n--- Running DFA Full-Context Masked Training (FCMT) Experiment ---")

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
