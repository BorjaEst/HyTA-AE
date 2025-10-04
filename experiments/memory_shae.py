"""State Hybrid Autoencoder (SHAE) experiment.

This module documents the refactor of the memory completion experiment into a
"state hybrid" variant that introduces a recurrent attractor in the decoder's
second hidden layer to recall a latent memory state from partial inputs.

Two-iterations scheme (as in `article/experiments/two-iterations.md`), adapted
for SHAE:

- Iteration 1 (state recall from partial cue): feed the partial observation
    (sensors[:, 0]) and run the decoder layer 2 with recurrent dynamics for a
    small number of steps until reaching a stable state (attractor). This latent
    state should approximate the closest stored memory compatible with the cue.
    No parameter updates here.
- Iteration 2 (training): form a fixed/completed input by mixing the first-pass
    reconstruction with observed pixels (via the mask), then run the model again
    (single pass) to compute local and global learning signals and update.

Architecture overview
---------------------
- Encoder: Two DFA layers provide local learning via output-error projection.
- Latent (DG-like): Growing linear layer with bounded activations to encourage
    sparse codes and progressive capacity expansion.
- Decoder: Two layers trained with hint/target-like local losses (HTL);
    the second decoder layer is recurrent to form an attractor for state recall.
- Output head: Linear projection back to the map space.

Training signals
----------------
- Reconstruction: BCE on the second pass output vs. the completion target.
- Encoder local: DFA-based local losses (error restricted to visible pixels).
- Decoder local: HTL-based alignment to encoder hidden states (second pass).
- Sparsity: HomeostaticActivityLoss on latent activations.

Notes and TODOs
---------------
- The current implementation still mirrors the previous memory-store HAE logic
    and uses targets in the first pass. The following tasks are planned:
    - TODO(SHAE): Make decoder.layer2 recurrent (simple fixed-point iteration or
        RNN block) and add a settling loop with parameters:
        - max_settle_steps (e.g., 8–32),
        - settle_tolerance (e.g., 1e-4 in L2 norm),
        - settle_damping (optional, 0<gamma<=1).
    - TODO(SHAE): Change `Autoencoder.forward` to take partial sensors input
        (sensors[:, 0]) and run the recurrent settle to obtain the attractor state
        before decoding the first reconstruction.
    - TODO(SHAE): Keep using mask channel (sensors[:, 1]) to compute completion
        targets and mask-aware DFA errors as in the HAE experiment.
    - TODO(SHAE): Add Pydantic config parameters to `Experiment` for the settle
        procedure (max steps, tolerance, damping) and logging switches.
    - TODO(SHAE): Add validation visualizations to inspect convergence traces of
        the recurrent layer per sample (optional).

All methods have PEP 257 docstrings and inline comments use the repository's
separator style. Logic is not modified in this refactor step; only
documentation/comments and TODOs are added.
"""

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
from torch.optim import Adam, Optimizer

from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.modules.loss import HomeostaticActivityLoss as SparsityLoss


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    """CLI-configurable hyperparameters for SHAE.

    Parameters are validated with Pydantic v2. Defaults are set to reproduce
    baseline runs. SHAE-specific parameters will be introduced in a later
    change without altering the public API of this module yet.

    Future additions (not yet implemented):
    - max_settle_steps: int — maximum recurrent iterations in decoder.layer2.
    - settle_tolerance: float — early-stop tolerance for fixed-point settling.
    - settle_damping: float — optional damping factor in [0, 1].
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Encoder and decoder components
    latent_units: PositiveInt = Field(default=2000, gt=0, description="Dimensionality of the latent code.")
    layer2_units: PositiveInt = Field(default=400, gt=0, description="Number of hidden units per layer.")
    layer1_units: PositiveInt = Field(default=5000, gt=0, description="Number of hidden units per layer.")
    output_shape: List[PositiveInt] = Field([25, 25], description="Dimensionality of the input and output.")

    # Data and augmentation parameters
    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    num_samples: PositiveInt = Field(default=4000, ge=100, le=10000, description="Number of samples to generate")
    mask_ratio: float = Field(default=0.65, ge=0.0, le=1.0, description="Fraction of spatial locations to mask")

    # Training Settings
    max_epochs: PositiveInt = Field(default=200, ge=1, description="Maximum training epochs")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field("mstore_hae", description="Experiment name")
    checkpoint_freq: PositiveInt = Field(default=5, ge=1, le=50, description="Checkpoint frequency")


# -------------------------------------------------------------------------------------------
class DFALayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_error: int):
        """Direct Feedback Alignment layer.

        - Projects output-space errors through a fixed random feedback matrix
          to obtain local targets.
        - Applies an activation (default: Identity) followed by tanh for
          bounded activity. Returned activations are detached to enforce
          locality.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            n_error: Dimensionality of the error vector (typically output dim).
        """
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.error_features = n_error
        self.register_buffer("activations", None)  # Starts without activation values
        self.register_buffer("fb_weight", torch.zeros(n_error, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute layer activations and detach them for local learning.

        Returns tanh(activation_fn(Wx + b)) with the result detached to stop
        gradient flow beyond the local learning rule.
        """
        currents = super().forward(*args, **kwargs)
        self.activations = torch.tanh(nn.functional.gelu(currents))
        return self.activations.detach()  # enforce locality

    @property
    def features(self) -> int:
        """Number of output features of the layer."""
        return self.out_features

    def feedback(self, error: Tensor) -> Tensor:
        """Compute DFA local loss from an output-space error.

        Args:
            error: Error tensor in output space, shape (B, n_error).

        Returns:
            Mean-squared error between current activations and their local
            DFA-derived targets.
        """
        delta = error.detach() @ self.fb_weight  # (batch_size, out_features)
        target = self.activations.detach() - delta  # same shape as activations
        return nn.functional.mse_loss(self.activations, target, reduction="mean")

    def reset_feedback(self) -> None:
        """Reinitialize the fixed feedback matrix with a uniform distribution."""
        limit = 1.0 / math.sqrt(self.error_features)
        nn.init.uniform_(self.fb_weight, -limit, limit)


# -------------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_inputs: int, n_h1: int, n_h2: int):
        """Two-layer DFA encoder.

        Args:
            n_inputs: Flattened input dimensionality (map size).
            n_h1: Hidden size for the first encoder layer.
            n_h2: Hidden size for the second encoder layer.
        """
        super().__init__()
        self.layer1 = DFALayer(n_inputs, n_h1, n_inputs)
        self.layer2 = DFALayer(n_h1, n_h2, n_inputs)

    def forward(self, x_incomplete: Tensor) -> List[Tensor]:
        """Forward pass returning hidden activations [h1, h2]."""
        h1 = self.layer1(x_incomplete)  # Use only channel 0 (obstacles)
        h2 = self.layer2(h1)
        return [h1, h2]

    def feedback(self, reconstruction_err: Tensor) -> Tensor:
        """Sum local DFA losses for both encoder layers.

        Args:
            reconstruction_err: Error in output space; typically masked to
                visible pixels before being flattened and passed here.
        """
        local_loss = torch.zeros(1, device=reconstruction_err.device)
        local_loss += self.layer2.feedback(reconstruction_err)
        local_loss += self.layer1.feedback(reconstruction_err)
        return local_loss


# -------------------------------------------------------------------------------------------
class HTLLayer(nn.Linear):
    def __init__(self, n_in: int, n_out: int, n_target: int):
        """Hint/Target-like layer for decoder-side local learning.

        This layer learns to reproduce a target (e.g., an encoder hidden state)
        given a context input, using an identity feedback mapping by default.

        Args:
            n_in: Number of input features.
            n_out: Number of output features.
            n_target: Target dimensionality; must equal n_out for identity FB.
        """
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.target_features = n_target
        self.register_buffer("fb_weight", torch.zeros(n_target, self.out_features))
        self.reset_feedback()  # Initialize weights properly

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass returning detached, bounded activations."""
        currents = super().forward(*args, **kwargs)
        activations = torch.tanh(nn.functional.gelu(currents))
        return activations.detach()  # enforce locality

    @property
    def features(self) -> int:
        """Number of output features of the layer."""
        return self.out_features

    def feedback(self, targets: Tensor, context: Tensor) -> Tensor:
        """Compute local HTL loss against provided targets.

        Args:
            targets: Teacher signals (e.g., encoder hidden states), shape
                (B, n_target).
            context: Pre-synaptic input used to produce the output.

        Returns:
            Mean-squared error between current output (no detach) and targets.
        """
        output = super().forward(context.detach())  # Detach pre-synaptic
        output = torch.tanh(nn.functional.gelu(output))
        targets = targets @ self.fb_weight  # (batch_size, out_features)
        return nn.functional.mse_loss(output, targets.detach(), reduction="mean")

    def reset_feedback(self) -> None:
        """Set the feedback matrix to identity (square-only for now)."""
        if self.fb_weight.shape[0] != self.out_features:
            raise ValueError("HTL only supports identity matrix for now.")
        nn.init.eye_(self.fb_weight)


# -------------------------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_h1: int, n_h2: int, n_latents: int):
        """Two-layer decoder trained with HTL local targets.

        SHAE note:
        - layer2 will become recurrent to implement attractor dynamics used in
          the first iteration (state recall). For now it remains feed-forward
          until the refactor lands.

        Args:
            n_h1: Size of the first decoder hidden layer.
            n_h2: Size of the second decoder hidden layer.
            n_latents: Latent dimensionality (decoder input size).
        """
        super().__init__()
        self.layer2 = HTLLayer(n_latents, n_h2, n_h2)
        self.layer1 = HTLLayer(n_h2, n_h1, n_h1)

    def forward(self, latent: Tensor) -> List[Tensor]:
        """Forward pass returning hidden activations [h1, h2].

        TODO(SHAE): When layer2 becomes recurrent, add an optional settle path
        here or expose a dedicated `settle` method that iterates layer2 until
        convergence. Keep this function simple and side-effect free.
        """
        h2 = self.layer2(latent)
        h1 = self.layer1(h2)
        return [h1, h2]

    def feedback(self, targets: List[Tensor], latent: Tensor) -> Tensor:
        """Sum local HTL losses for both decoder layers.

        Args:
            targets: Encoder hidden states [h1_enc, h2_enc] used as targets.
            latent: Latent codes used as decoder input.

        Returns:
            Local MSE loss accumulated over both decoder layers.
        """
        local_loss = torch.zeros(1, device=latent.device)
        local_loss += self.layer2.feedback(targets[1], context=latent)
        local_loss += self.layer1.feedback(targets[0], context=targets[1])
        return local_loss


# -------------------------------------------------------------------------------------------
class DGLayer(nn.Linear):
    def __init__(self, n_in: int, n_init: int, n_max: int):
        """DG-like growing latent layer with masked outputs.

        Args:
            n_in: Input feature count.
            n_init: Initial active units.
            n_max: Maximum capacity (total output features).
        """
        super().__init__(in_features=n_in, out_features=n_max, bias=True)
        self.register_buffer("output_mask", torch.zeros(n_max, dtype=torch.float32))
        self.grow(n_init)  # Initialize with n_init active units

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Apply output mask and bounded activation to produce sparse codes."""
        out = super().forward(*args, **kwargs)
        out = out * self.output_mask  # Masked units are zeroed; grads do not flow to their rows
        return torch.tanh(torch.relu(out))  # Using ReLU followed by Tanh for bounded activations

    @property
    def out_active(self) -> int:
        """Current number of active output units."""
        return sum(self.output_mask).int().item()

    @property
    def capacity(self) -> int:
        """Total capacity (maximum number of output units)."""
        return self.weight.shape[0]

    @torch.no_grad()
    def grow(self, n: int) -> None:
        """Activate the next n output units (capped by capacity).

        Note: The method assumes moderate growth per epoch. If long training
        runs are used, ensure not to exceed `capacity`.
        """
        self.output_mask[: self.out_active + n] = 1.0


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, output_shape: List[int], n_layer1: int, n_layer2: int, n_latent: int):
        """State Hybrid Autoencoder for map completion.

        Args:
            output_shape: Spatial shape of the maps (H, W).
            n_layer1: Size of decoder first hidden layer / encoder mirror.
            n_layer2: Size of encoder second hidden layer / decoder mirror.
            n_latent: Maximum latent capacity for the growing DG layer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["trainer"])
        self.automatic_optimization = False
        self.output_shape = output_shape
        n_output = math.prod(output_shape)

        # Initialize encoder and decoder (decoder.layer2 will be recurrent in SHAE)
        self.encoder = Encoder(n_output, n_layer1, n_layer2)
        self.latent = DGLayer(n_layer2, n_latent // 10, n_latent)
        self.decoder = Decoder(n_layer1, n_layer2, n_latent)
        self.output = nn.Linear(n_layer1, n_output)

        # Loss functions
        self.reconstruction_loss = nn.BCELoss(reduction="mean")
        self.sparsity_loss = SparsityLoss(target_rate=0.05, min_active=8)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        """Configure Adam with per-module learning rates.

        Lower learning rate on encoder and latent stabilizes local learning and
        sparsity dynamics.
        """
        optimizer_parameters = [
            {"params": self.encoder.parameters(), "lr": 1e-5},
            {"params": self.latent.parameters(), "lr": 1e-8},
            {"params": self.decoder.parameters(), "lr": 1e-4},
            {"params": self.output.parameters(), "lr": 1e-4},
        ]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def _encode(self, inputs: Tensor) -> Tensor:
        """Private helper: encode inputs and pass through latent grower.

        SHAE note: In the future, the first iteration will not encode targets
        but rather run a partial-cue path that uses decoder.layer2 recurrence to
        retrieve the latent memory state. This method remains as-is for now.
        """
        encoder_signals = self.encoder(flatten(inputs, start_dim=1))
        return self.latent(encoder_signals[-1])

    @torch.inference_mode()
    def encode(self, sensors: Tensor) -> Tensor:
        """Encode sensors into latent space using the DFA-based encoder."""
        return self._encode(sensors)

    def _decode(self, latent: Tensor) -> Tensor:
        """Private helper: decode latent codes to probability maps in [0, 1].

        TODO(SHAE): When the recurrent settle is available, decoding after the
        attractor will be applied here (no change required to the API).
        """
        decoder_signals = self.decoder(latent)
        logits = unflatten(self.output(decoder_signals[0]), 1, self.output_shape)
        return torch.sigmoid(logits)

    @torch.inference_mode()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent codes into map space using the HTL-based decoder."""
        return self._decode(latent)

    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Iteration 1 (current placeholder): reconstruction/latent from targets.

        Args:
            batch: Tuple of (sensors, targets). Targets are used here but will
                be replaced by partial sensors in the SHAE refactor.

        Returns:
            reconstruction: Probability map in [0, 1].
            latent: Latent activations (pre-detach) from the DG layer.
        """
        sensors, _targets = batch
        latent = self._encode(sensors[:, 0])  # Use only channel 0 (obstacles)
        reconstruction = self._decode(latent.detach())
        return reconstruction, latent

    # -----------------------------------------------------------------------------------
    def feedback(self, first_prediction: Tensor, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Second pass: compute local/global losses and return total.

        Forms the completion input by mixing the first pass prediction with the
        observed pixels, then applies:
        - Encoder DFA losses (masked error),
        - Latent sparsity loss,
        - Decoder HTL losses (targets from encoder on the completion),
        - Reconstruction BCE on the completion target.

        SHAE note: The error masking and completion rule remain the same. The
        only change in SHAE is how `first_prediction` is obtained (from partial
        cue + attractor settling), which will be handled in `forward` later.
        """
        sensors, targets = batch
        x_incomplete, mask = sensors[:, 0], sensors[:, 1]

        # Create completion target for encoder/decoder feedback paths
        completion = x_incomplete * mask + first_prediction * (1 - mask)
        error = first_prediction * mask - x_incomplete  # DFA feedback error only on visible pixels

        # Create targets and contexts for feedback paths
        decoder_targets = self.encoder(flatten(completion, start_dim=1))
        latent = self.latent(decoder_targets[-1])  # detached by DFALayer

        # Second pass to update activations and reconstruction
        reconstruction = self._decode(latent.detach())

        # Train the autoencoder layers with dfa, sparsity, htl and standard loss
        local_loss = self.encoder.feedback(flatten(error, start_dim=1))
        local_loss += self.sparsity_loss(latent)
        local_loss += self.decoder.feedback(decoder_targets, latent.detach())
        local_loss += self.reconstruction_loss(reconstruction, completion.detach())

        return local_loss

    # -----------------------------------------------------------------------------------
    def on_train_epoch_start(self) -> None:
        """Grow a fixed number of latent units at the start of each epoch."""
        self.latent.grow(8)  # Grow latent units per epoch till maximum capacity

    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        """Manual optimization step following the two-iterations procedure.

        TODO(SHAE): After implementing the recurrent settle in `forward`, this
        step will naturally use the attractor-based `first_prediction`.
        """
        # First we produce the reconstruction, we ignore the latent obtained from targets
        with torch.no_grad():
            first_prediction, first_latent = self(batch)

        # Now we train using feedback connections
        self.optimizers().zero_grad()
        local_loss = self.feedback(first_prediction, batch)
        self.manual_backward(local_loss)
        self.optimizers().step()

        # Logging metrics: use first prediction to match validation
        self.log_metrics("train", first_prediction, batch[1], first_latent)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        """Validation: mirror the first pass and log alignment diagnostics.

        TODO(SHAE): Add optional diagnostics of recurrent settling convergence
        (iteration count to tolerance, delta norms) once the settle loop is in.
        """
        reconstruction, latent = self(batch)
        self.log_metrics("val", reconstruction, batch[1], latent)
        encoder_signals = self.encoder(flatten(reconstruction, start_dim=1))
        decoder_signals = self.decoder(latent)
        self.log_hidden("val", "1", decoder_signals[0], encoder_signals[0])
        self.log_hidden("val", "2", decoder_signals[1], encoder_signals[1])

    # -----------------------------------------------------------------------------------
    def log_metrics(self, stage: str, reconstruction: Tensor, targets: Tensor, latent: Tensor) -> None:
        """Log scalar metrics common to train/val stages."""
        x_mseloss = nn.functional.mse_loss(reconstruction, targets, reduction="mean")
        self.log(f"{stage}/reconstruction_loss", x_mseloss, prog_bar=True, on_step=False, on_epoch=True)
        sparsity_rate = (latent.abs() < 0.01).float().mean()
        self.log(f"{stage}/sparsity_rate", sparsity_rate, prog_bar=True, on_step=False, on_epoch=True)

    def log_hidden(self, stage: str, i: int, hi_decoder: Tensor, hi_encoder: Tensor) -> None:
        """Log hidden-state alignment losses for decoder vs. encoder layers."""
        h1_mseloss = nn.functional.mse_loss(hi_decoder, hi_encoder.detach(), reduction="mean")
        self.log(f"{stage}/h{i}_mseloss", h1_mseloss, prog_bar=True, on_step=False, on_epoch=True)


# -------------------------------------------------------------------------------------------
def gen_figures(model: Autoencoder, datamodule: BaseDataModule) -> None:
    """Generate reconstruction and sparsity figures from model outputs.

    Uses a single batch from the test set to:
    - Visualize target vs. output reconstructions,
    - Inspect latent sparsity statistics,
    - Probe decoder selectivity with one-hot latent codes.

    TODO(SHAE): Add a figure showing the effect of recurrent settling from a
    partial cue, e.g., plot reconstruction over settle steps or the evolution
    of the decoder.layer2 state norms.
    """
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
    datamodule_params = DataModuleParams(num_samples=experiment.num_samples)
    datamodule = BaseDataModule(data_gen, datamodule_params)

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
