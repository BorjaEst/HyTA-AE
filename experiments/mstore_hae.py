import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, flatten, nn, unflatten
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Encoder and decoder components
    latent_units: PositiveInt = Field(default=2048, gt=0, description="Dimensionality of the latent code.")
    layer2_units: PositiveInt = Field(default=512, gt=0, description="Number of hidden units per layer.")
    layer1_units: PositiveInt = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[PositiveInt] = Field([25, 25], description="Dimensionality of the input and output.")

    # Data and augmentation parameters
    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    mask_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")

    # Training Settings
    max_epochs: PositiveInt = Field(default=200, ge=1, le=1000, description="Maximum training epochs")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field("mstore_hae", description="Experiment name")
    checkpoint_freq: PositiveInt = Field(default=5, ge=1, le=50, description="Checkpoint frequency")


# -------------------------------------------------------------------------------------------
class DFALayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_error: int, activation_fn: Optional[nn.Module] = None):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.error_features = n_error
        self.activation_fn = activation_fn or nn.Identity()
        self.register_buffer("activations", None)  # Starts without activation values
        self.register_buffer("fb_weight", torch.zeros(n_error, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = self.activation_fn(currents)
        return self.activations.detach()  # enforce locality

    @property
    def features(self) -> int:
        """Number of output features of the layer."""
        return self.out_features

    def feedback(self, error: Tensor) -> None:
        delta = error.detach() @ self.fb_weight  # (batch_size, out_features)
        torch.autograd.backward(self.activations, delta)
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def reset_feedback(self) -> None:
        limit = 1.0 / math.sqrt(self.error_features)
        nn.init.uniform_(self.fb_weight, -limit, limit)


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int):
        super().__init__()
        self.layer1 = DFALayer(n_inputs, n_h1, n_inputs, nn.GELU())
        self.layer2 = DFALayer(n_h1, n_h2, n_inputs, nn.GELU())

    def forward(self, x_incomplete: Tensor) -> List[Tensor]:
        h1 = self.layer1(x_incomplete)  # Use only channel 0 (obstacles)
        h2 = self.layer2(h1)
        return [h1, h2]

    def feedback(self, reconstruction_err: Tensor) -> None:
        self.layer2.feedback(reconstruction_err)
        self.layer1.feedback(reconstruction_err)


# -------------------------------------------------------------------------------------------
class HTLLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_target: int, activation_fn: Optional[nn.Module] = None):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.target_features = n_target
        self.activation_fn = activation_fn or nn.Identity()
        self.register_buffer("fb_weight", torch.zeros(n_target, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        activations = self.activation_fn(currents)
        return activations.detach()  # enforce locality

    @property
    def features(self) -> int:
        """Number of output features of the layer."""
        return self.out_features

    def feedback(self, targets: Tensor, context: Tensor) -> None:
        output = super().forward(context.detach())  # Detach pre-synaptic
        output = self.activation_fn(output)
        targets = targets @ self.fb_weight  # (batch_size, out_features)
        F.mse_loss(output, targets.detach(), reduction="mean").backward()
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def reset_feedback(self) -> None:
        if self.fb_weight.shape[0] != self.out_features:
            raise ValueError("HTL only supports identity matrix for now.")
        nn.init.eye_(self.fb_weight)


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_h1: int, n_h2: int, n_latents: int):
        super().__init__()
        self.layer2 = HTLLayer(n_latents, n_h2, n_h2, nn.GELU())
        self.layer1 = HTLLayer(n_h2, n_h1, n_h1, nn.GELU())

    def forward(self, latent: Tensor) -> List[Tensor]:
        h2 = self.layer2(latent)
        h1 = self.layer1(h2)
        return [h1, h2]

    def feedback(self, encoder: Encoder, latent: Tensor) -> None:
        self.layer2.feedback(encoder.layer2.activations, context=latent)
        self.layer1.feedback(encoder.layer1.activations, context=encoder.layer2.activations)


# -------------------------------------------------------------------------------------------
class DGLayer(nn.Linear):
    def __init__(self, n_in: int, n_init: int, n_max: int, activation_fn: Optional[nn.Module] = None):
        super().__init__(in_features=n_in, out_features=n_max, bias=True)
        self.activation_fn = activation_fn or nn.ReLU()
        self.register_buffer("output_mask", torch.zeros(n_max, dtype=torch.float32))
        self.grow(n_init)  # Initialize with n_init active units

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        out = super().forward(*args, **kwargs)
        out = out * self.output_mask  # Masked units are zeroed; grads do not flow to their rows
        return self.activation_fn(out)

    @property
    def out_active(self) -> int:
        return sum(self.output_mask).int().item()

    @property
    def capacity(self) -> int:
        return self.weight.shape[0]

    @torch.no_grad()
    def grow(self, n: int) -> None:
        self.output_mask[: self.out_active + n] = 1.0


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, output_shape: List[int], n_layer1: int, n_layer2: int, n_latent: int):
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.automatic_optimization = False
        self.output_shape = output_shape
        n_output = math.prod(output_shape)

        # Initialize encoder and decoder with DFA layers
        self.encoder = Encoder(n_output, n_layer1, n_layer2)
        self.latent = DGLayer(n_layer2, n_latent // 10, n_latent)
        self.decoder = Decoder(n_layer1, n_layer2, n_latent)
        self.output = nn.Linear(n_layer1, n_output)

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(center=True)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder.parameters(), "lr": 1e-5},
            {"params": self.latent.parameters(), "lr": 1e-8},
            {"params": self.decoder.parameters(), "lr": 1e-4},
            {"params": self.output.parameters(), "lr": 1e-4},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def _encode(self, inputs: Tensor) -> Tensor:
        encoder_signals = self.encoder(flatten(inputs, start_dim=1))
        return self.latent(encoder_signals[-1])

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        """Encode sensors into latent space using the DFA-based encoder."""
        return self._encode(sensors)

    def _decode(self, latent: Tensor) -> Tensor:
        decoder_signals = self.decoder(latent)
        logits = unflatten(self.output(decoder_signals[0]), 1, self.output_shape)
        return F.sigmoid(logits)

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent codes into map space using the HTL-based decoder."""
        return self._decode(latent)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        sensors, _targets = batch
        latent = self._encode(sensors[:, 0])  # Use channel 0 with masked targets
        reconstruction = self._decode(latent.detach())
        return reconstruction, latent

    # -----------------------------------------------------------------------------------
    def feedback(self, first_prediction: Tensor, batch: Tensor) -> None:
        sensors, targets = batch
        x_incomplete, mask = sensors[:, 0], sensors[:, 1]

        # Create completion target for encoder/decoder feedback paths
        completion = x_incomplete * mask + first_prediction * (1 - mask)
        error = first_prediction * mask - x_incomplete  # DFA feedback error only on visible pixels

        # Create targets and contexts for feedback paths
        targets = self.encoder(flatten(completion, start_dim=1))
        latent = self.latent(targets[-1])  # detached by DFALayer

        # Second pass to update activations and reconstruction
        reconstruction = self._decode(latent.detach())

        # Train the autoencoder layers with dfa, sparsity, htl and standard loss
        self.encoder.feedback(flatten(error, start_dim=1))
        self.sparsity_loss(latent).backward()
        self.decoder.feedback(self.encoder, latent.detach())
        self.reconstruction_loss(reconstruction, completion.detach()).backward()

    # -----------------------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        self.latent.grow(10)  # Grow latent units per epoch till maximum capacity

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        # First we produce the reconstruction, we ignore the latent obtained from targets
        with torch.no_grad():
            first_prediction, _latent_from_target = self(batch)

        # Now we train using feedback connections
        self.optimizers().zero_grad()
        self.feedback(first_prediction, batch)
        self.optimizers().step()

    # -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tensor, batch_idx: int) -> List[Tensor]:
        reconstruction, latent = self(batch)
        self.log_metrics_x(reconstruction, batch[1])  # targets to measure reconstruction loss
        self.log_metrics_z(latent)
        encoder_signals = self.encoder(flatten(reconstruction, start_dim=1))
        decoder_signals = self.decoder(latent)
        self.log_metrics_hi("1", decoder_signals[0], encoder_signals[0])
        self.log_metrics_hi("2", decoder_signals[1], encoder_signals[1])

    def log_metrics_x(self, reconstruction: Tensor, targets: Tensor) -> None:
        x_mseloss = F.mse_loss(reconstruction, targets, reduction="mean")
        self.log("val/reconstruction_loss", x_mseloss, prog_bar=True, on_step=False, on_epoch=True)

    def log_metrics_z(self, latent: Tensor) -> None:
        sparsity_rate = (latent.abs() < 0.01).float().mean()
        self.log("val/sparsity_rate", sparsity_rate, prog_bar=True, on_step=False, on_epoch=True)

    def log_metrics_hi(self, i: int, h1_decoder: Tensor, h1_encoder: Tensor) -> None:
        h1_mseloss = F.mse_loss(h1_decoder, h1_encoder.detach(), reduction="mean")
        self.log(f"val/h{i}_mseloss", h1_mseloss, prog_bar=True, on_step=False, on_epoch=True)


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
    print(f"\n--- Running Data Completion with Feedback Experiment ---")

    # Initialize experiment configuration
    experiment = Experiment()
    composition_params = ComposeParams(mask_ratio=experiment.mask_ratio)
    augmentation = Augmentation(composition_params)
    data_gen = DataGenerator(experiment.data, augmentation)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)

    # Initialize model with specified architecture
    model = Autoencoder(
        output_shape=experiment.output_shape,
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
        gen_figures(model, datamodule)
