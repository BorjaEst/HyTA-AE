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
from ehc_sn.core import ann
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.models.ann.sparse_autoencoder import Autoencoder as Teacher
from ehc_sn.models.ann.sparse_autoencoder import ModelParams as TeacherParams
from ehc_sn.modules.loss import GramianOrthogonalityLoss as SparsityLoss


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """Configuration settings for the hybrid autoencoder experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Autoencoder model configuration
    layer1_units: PositiveInt = Field(default=1024, gt=0, description="Number of hidden units per layer.")
    output_shape: List[PositiveInt] = Field([25, 25], description="Dimensionality of the input and output.")
    teacher_path: str = Field(default="models/overcomplete_autoencoders/backprop_2048.ckpt")

    # Data and augmentation parameters
    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    mask_ratio: float = Field(default=0.4, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")

    # Training Settings
    max_epochs: PositiveInt = Field(default=200, ge=1, le=1000, description="Maximum training epochs")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field("teacher_1layers", description="Experiment name")
    checkpoint_freq: PositiveInt = Field(default=5, ge=1, le=50, description="Checkpoint frequency")


# -------------------------------------------------------------------------------------------
class DFALayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_error: int, activation_fn: Optional[nn.Module] = None):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("currents", None)  # Starts without current values
        self.register_buffer("activations", None)  # Starts without activation values
        self.activation_fn = activation_fn or nn.Identity()
        self.register_buffer("fb_weight", torch.zeros(n_error, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        self.currents = super().forward(*args, **kwargs)
        self.activations = self.activation_fn(self.currents)
        return self.activations.detach()  # enforce locality

    @property
    def features(self) -> int:
        """Number of output features of the layer."""
        return self.out_features

    def feedback(self, error: Tensor) -> None:
        delta = error.detach() @ self.fb_weight  # (batch_size, out_features)
        # torch.autograd.backward(self.currents, delta)
        torch.autograd.backward(self.activations, delta)

    def reset_feedback(self) -> None:
        limit = 1.0 / math.sqrt(self.fb_weight.shape[0])
        torch.nn.init.uniform_(self.fb_weight, -limit, limit)


# -------------------------------------------------------------------------------------------
class HTLLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_target: int, activation_fn: Optional[nn.Module] = None):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("currents", None)  # Starts without current values
        self.register_buffer("activations", None)  # Starts without activation values
        self.activation_fn = activation_fn or nn.Identity()
        self.register_buffer("fb_weight", torch.zeros(n_target, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        self.currents = super().forward(*args, **kwargs)
        self.activations = self.activation_fn(self.currents)
        return self.activations.detach()  # enforce locality

    @property
    def features(self) -> int:
        """Number of output features of the layer."""
        return self.out_features

    def feedback(self, targets: Tensor) -> None:
        # F.mse_loss(self.currents, target.detach(), reduction="mean").backward()
        targets = targets @ self.fb_weight  # (batch_size, out_features)
        F.mse_loss(self.activations, targets.detach(), reduction="mean").backward()

    def reset_feedback(self) -> None:
        if self.fb_weight.shape[0] != self.out_features:
            raise ValueError("HTL only supports identity matrix for now.")
        torch.nn.init.eye_(self.fb_weight)


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, output_shape: List[int], n_layer1: int, teacher: nn.Module):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.automatic_optimization = False
        self.output_shape = output_shape
        n_output = math.prod(output_shape)
        n_layer2 = teacher.encoder.layer2.out_features

        # Initialize encoder and decoder with DFA and HTL layers
        self.teacher = teacher
        self.encoder_layer1 = DFALayer(n_output, n_layer1, n_output, nn.GELU())
        self.decoder_layer1 = HTLLayer(n_layer2, n_layer1, n_layer1, nn.GELU())
        self.output_layer = nn.Linear(n_layer1, n_output)  # Output layer (no HTL)

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(center=True)
        self.teacher.eval()

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [
            {"params": self.encoder_layer1.parameters(), "lr": 1e-4},
            {"params": self.decoder_layer1.parameters(), "lr": 1e-3},
            {"params": self.output_layer.parameters(), "lr": 1e-3},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_h2(self, targets: Tensor) -> Tensor:
        latent = self.teacher.encode(targets)
        return self.teacher.decoder.layer2(latent)

    # -----------------------------------------------------------------------------------
    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        _sensors, targets = batch
        h2_teacher = self.sample_h2(flatten(targets, start_dim=1))  # Detach by @torch.no_grad
        h1 = self.decoder_layer1(h2_teacher)  # Detached by HTLLayer
        logits = self.output_layer(h1)
        reconstruction = torch.sigmoid(logits)
        return unflatten(reconstruction, 1, targets.shape[1:]), h2_teacher

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        h1 = self.encoder_layer1(flatten(sensors, start_dim=1))
        return h1

    @torch.inference_mode()
    def decode(self, h1: Tensor) -> Tensor:
        reconstruction = unflatten(self.output_layer(h1), 1, self.output_shape)
        return torch.sigmoid(reconstruction)

    # -----------------------------------------------------------------------------------
    def feedback(self, reconstruction: Tensor, batch: Tensor) -> None:
        sensors, targets = batch
        x_incomplete, mask = sensors[:, 0], sensors[:, 1]
        reconstruction = torch.nan_to_num(reconstruction, nan=0.5, posinf=1.0, neginf=0.0)
        reconstruction = reconstruction.clamp_(0.0, 1.0)

        # Create completion target for encoder/decoder feedback paths
        completion = x_incomplete * mask + reconstruction * (1 - mask)

        # Encoder DFA feedback uses error only on visible pixels
        error = reconstruction * mask - x_incomplete
        h1_target = self.encoder_layer1(flatten(completion, start_dim=1).detach())
        self.encoder_layer1.feedback(error.flatten(start_dim=1))
        self.decoder_layer1.feedback(h1_target.detach())

        # Loss propagation for output layer
        loss_output = nn.BCELoss(reduction="mean")(reconstruction, completion.detach())
        loss_output.backward()

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        reconstruction, _h2_teacher = self(batch)
        self.feedback(reconstruction, batch)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizers().step()

    # -----------------------------------------------------------------------------------
    def validation_step(self, batch: Tensor, batch_idx: int) -> List[Tensor]:
        sensors, targets = batch
        reconstruction, _h2_teacher = self(batch)
        reconstruction_loss = nn.MSELoss(reduction="mean")(reconstruction, targets)
        self.log("val/reconstruction_loss", reconstruction_loss, prog_bar=True)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule) -> None:
    """Generate reconstruction and sparsity figures from model outputs."""
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    _, targets = batch = next(iter(test_dataloader))
    with torch.inference_mode():
        outputs, h2_teacher = model(batch)

    # Figure 1: Reconstruction map comparing inputs and outputs
    fig_reconstruction = ReconstructionMapFigure()
    _ = fig_reconstruction.plot(targets, outputs)
    plt.show()

    h1_dim = model.encoder_layer1.out_features
    h1 = torch.eye(h1_dim)[:18]  # One-hot encoding for each unit

    # Generate decoder outputs for one-hot latents
    reconstructions = model.decode(h1)

    # Figure 2: Decoder montage showing individual latent unit reconstructions
    decoder_montage_figure = DecoderMontageFigure()
    _ = decoder_montage_figure.plot(h1, reconstructions)
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

    # Load teacher model and its parameters
    teacher_ckpt = torch.load(experiment.teacher_path)
    teacher_params = TeacherParams(latent_units=2048)
    teacher = Teacher(teacher_params, trainer=None)

    # Filter only weight/bias keys to ignore obsolete buffers
    state_dict = teacher_ckpt["state_dict"]
    filtered = {k: v for k, v in state_dict.items() if k.endswith(".weight") or k.endswith(".bias")}
    teacher.load_state_dict(filtered, strict=True)

    # Initialize model with specified architecture
    model = Autoencoder(experiment.output_shape, experiment.layer1_units, teacher)

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
