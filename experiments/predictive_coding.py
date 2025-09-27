"""Predictive-coding experiment with strictly local learning placeholders.

This module implements a minimal, analysis-friendly predictive-coding setup for an
autoencoder-style decoder with explicit separation between inference and local learning.
The code is intentionally simple and leaves some local learning rules as TODOs to enable
focused experiments and documentation.

Mathematical correspondence (see article/predictive-coding/README.md):

- Variables and shapes (batch dimension implicit):
  - x in R^D: ground-truth target ("sensors" are ignored for this study)
  - h2 in R^{H2}: cached hidden state from a frozen teacher encoder (top-down drive)
  - h1_hat in R^{H1}: corrected hidden state at the first decoder layer
  - W_d1: first decoder layer mapping h2 -> h1_base
  - W_out: output layer mapping h1_hat -> x_hat (borrowed from the teacher)
  - W_e: error->hidden mapping; implemented here as EncoderLayer (nn.Linear)
  - B: correction mixing (combination) matrix applied at decoder h1 (identity by default)
  - F: feedback projection buffer for W_e (initialized as identity or random)

Core one-step relations implemented by forward paths:

  h1_base = W_d1 h2
  h_e     = f(W_e e_0)
  h1_hat  = h1_base + B h_e
  x_hat   = W_out h1_hat
  e_0     = x - x_hat

Local stability (analysis-only; not enforced by code):
  J = I - W_out B D W_e, with D = diag(f'(0)). A sufficient local contraction
  condition around the perfect fixed point e_0=0 is rho(J) < 1. This file does not
  alter learning based on J; it just exposes the operators in a way that enables
  external stability probes and reporting.

Important: This script avoids transposes of forward weights for the corrective path.
All updates to W_e (EncoderLayer) are left as TODOs to preserve strict locality
exploration. The DecoderLayer contains a local target-matching loss to "absorb"
the correction into W_d1, as described in the article.
"""

import math
from typing import List, Literal, Optional, Tuple

import torch
from lightning import pytorch as pl
from torch import Tensor, flatten, nn
from torch.optim import Optimizer

from ehc_sn.models.ann.sparse_autoencoder import Autoencoder as Teacher
from ehc_sn.models.ann.sparse_autoencoder import ModelParams as TeacherParams

# -------------------------------------------------------------------------------------------
FEEDBACK_MODE: Literal["random", "identity"] = "random"
COMBINATION_MODE: Literal["identity", "random"] = "identity"
ACTIVATION_FN: bool = True  # whether to use non-linear activations
PERFECT_DECODER_INIT: bool = True  # whether to init decoder with teacher weights
TRAIN_ENCODER_LAYER: bool = True  # whether to train the first encoder layer
TRAIN_DECODER_LAYER: bool = True  # whether to train the first decoder layer
TRAIN_OUTPUT_LAYER: bool = False  # whether to train the final output layer
INFERENCE_STEPS: int = 5  # number of inference steps


# -------------------------------------------------------------------------------------------
class EncoderLayer(nn.Linear):
    """Error→hidden map W_e with a local feedback projection F.

    This layer represents the strictly local corrective pathway from output error e_0
    to the hidden correction h_e used at the first decoder layer. It holds a buffer
    ``F`` that can serve as an independent (random/identity) feedback projection and
    caches the post-activation ``activations`` to support local rules.

    Notation alignment with the docs:
    - W_e: this module's trainable weights (nn.Linear)
    - f(·): GELU or Identity (controlled by ACTIVATION_FN)
    - h_e = f(W_e e_0)
    - F: optional fixed feedback projection (not used in forward, reserved for local updates)

    Notes on locality and stability:
    - Local surrogate updates for W_e (see article) use only e_0, unit-local slopes,
      an external teaching signal F' e_0, and optional penalties; we keep the method
      placeholders here to enable controlled experiments without changing behavior.
    - Stability near e_0≈0 is governed by the composite J = I - W_out B D W_e (analysis-only).
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, **kwargs) -> None:
        super().__init__(in_features, out_features, bias=False, **kwargs)
        self.register_buffer("F", torch.empty(out_features, in_features))  # feedback projection
        self.activation = nn.GELU() if ACTIVATION_FN else nn.Identity()
        self.register_buffer("activations", None)  # h_e = B f
        self.init_feedback()  # default init

    def feedback(self, error: Tensor) -> None:
        """Placeholder to receive a local teaching signal for W_e updates.

        In the strict-locality recipe, one would form a surrogate hidden error
        ``tilde_delta_h = F' e_0`` and combine it with the unitwise slope to
        produce a local update on W_e. This experiment keeps learning for W_e
        as a TODO to avoid changing behavior while documenting the logic.
        """
        pass  # TODO (strictly local W_e update lives here if/when enabled)

    def forward(self, error: Tensor) -> Tensor:
        """Map output error to hidden correction: u=W_e e_0, h_e=f(u).

        Parameters
        ----------
        error: Tensor
            Current reconstruction error e_0 in R^D (flattened per sample).

        Returns
        -------
        Tensor
            Hidden correction activations h_e in R^{H1} to be mixed at decoder h1.
        """
        currents = super().forward(error.detach())
        self.activations = self.activation(currents)
        return self.activations

    def update(self) -> None:
        """Placeholder for a strictly local W_e learning rule.

        See article/predictive-coding/README.md for the recommended local update
        that uses only local signals and a fixed projection; intentionally not
        implemented to keep the experiment's behavior unchanged.
        """
        pass  # TODO (local W_e rule)

    def init_feedback(self) -> None:
        """Initialize the feedback projection F (identity or random).

        This F is distinct from forward weights and is intended as a fixed,
        independently drawn projection used by local learning surrogates (if enabled).
        """
        if FEEDBACK_MODE == "identity":
            self.F.copy_(torch.eye(*self.F.shape, device=self.F.device, dtype=self.F.dtype))
        elif FEEDBACK_MODE == "random":
            torch.nn.init.kaiming_uniform_(self.F, a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown feedback mode: {FEEDBACK_MODE}")


# -------------------------------------------------------------------------------------------
class DecoderLayer(nn.Linear):
    """First decoder layer W_d1 with local absorption of hidden corrections.

    This layer implements the base hidden drive h1_base = W_d1 h2 and adds a prepared
    correction at the pre-activation level using a mixing matrix B:

      currents = W_d1 h2 + (B h_e)  [when corrections are provided]
      h1_hat  = f(currents)

    The ``update`` method performs a strictly local target-matching step to absorb
    the current correction into W_d1, using the quadratic loss
    1/2 || W_d1 h2 - (W_d1 h2 - B h_e) ||^2 at the pre-activation level. This matches
    the "absorption" rule described in the docs and uses only local pre/post signals.
    """

    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.register_buffer("B", torch.empty(out_features, out_features))  # combination / modulation
        self.activation = nn.GELU() if ACTIVATION_FN else nn.Identity()
        self.register_buffer("corrections", None)  # (B, H1) = B h_e
        self.register_buffer("currents", None)  # pre-activations u = W_d h2
        self.init_combination()  # default init
        self.loss_fn = nn.MSELoss(reduction="mean")

    def feedback(self, corrections: Tensor) -> None:
        """Prepare top-down pre-activation corrections as B h_e.

        Parameters
        ----------
        corrections: Tensor
            Hidden correction h_e in R^{H1} (batch x H1). Will be mixed by B and
            added to currents on the next forward call.
        """
        # Prepare the correction to be applied at next topdown pass
        self.corrections = corrections.detach() @ self.B.T

    def forward(self, input: Tensor) -> Tensor:
        """Compute h1_hat = f(W_d1 h2 + B h_e) with cached pre-activations.

        Notes
        -----
        - The corrections are additive at the pre-activation level (currents), matching
          the mathematical sketch. If no correction is queued, the base path is used.
        - We cache currents for the local absorption loss in ``update``.
        """
        # Standard forward pass with possible corrections
        self.currents = super().forward(input.detach())
        if self.corrections is not None:
            self.currents += self.corrections
        return self.activation(self.currents)

    def update(self) -> None:
        """Locally absorb the last correction into W_d1 via pre-activation matching.

        Implements the quadratic loss L = 1/2 || u - (u - B h_e) ||^2 at the level of
        pre-activations u, whose gradient step pushes W_d1 so that future predictions
        rely less on the explicit correction. This operation is strictly local: it uses
        cached ``currents`` (post-synaptic) and the prepared correction (also cached).
        """
        # Update weights W_d1 to push u toward u* = u - B h_e
        local_loss = self.loss_fn(self.currents.detach(), self.currents - self.corrections)
        local_loss.backward()
        self.corrections = None  # reset after use

    def init_combination(self) -> None:
        """Initialize the mixing matrix B (identity or random)."""
        if COMBINATION_MODE == "identity":
            self.B.copy_(torch.eye(*self.B.shape, device=self.B.device, dtype=self.B.dtype))
        elif COMBINATION_MODE == "random":
            torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown combination mode: {COMBINATION_MODE}")


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    """Predictive-coding experiment with explicit inference and local learning split.

    High-level flow per training step (outer loop):
    - Cache h2 from a frozen teacher encoder for the given target x.
    - Iterate a small number of inference steps:
        1) Compute a hidden correction h_e = f(W_e e_0) via ``EncoderLayer``.
        2) Queue the pre-activation correction B h_e into ``DecoderLayer``.
        3) Top-down pass to obtain x_hat and compute e_0 = x - x_hat.
        4) Apply local absorption update on W_d1 (``DecoderLayer.update``).
        5) Optionally train the output layer using the reconstruction loss.

    Learning for W_e is intentionally left as a TODO to keep the experiment invariant
    while we document math, signals, and stability considerations.
    """

    def __init__(self, teacher: nn.Module) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.config = params = teacher.config
        self.automatic_optimization = False
        teacher.eval()

        # Dimensions
        n_h1, n_h2 = params.layer1_units, params.layer2_units
        n_sensors = math.prod(params.output_shape)

        # Model components (only encoder_layer1 is trained)
        self.teacher = teacher
        self.encoder_layer1 = EncoderLayer(n_sensors, n_h1, bias=False)  # ensure no constant drive
        self.decoder_layer1 = DecoderLayer(n_h2, n_h1, bias=True)
        self.decoder_output = teacher.decoder.output

        # Initialize weights
        if PERFECT_DECODER_INIT:  # copy weights from teacher
            self.decoder_layer1.weight.data.copy_(teacher.decoder.layer1.synapses.weight.data)
            self.decoder_layer1.bias.data.copy_(teacher.decoder.layer1.synapses.bias.data)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        """Configure a single Adam optimizer with per-group learning rates.

        Notes
        -----
        - TRAIN_* flags control which parameter groups are effectively trained.
        - We rely on Lightning's manual optimization to orchestrate the step timing
          across the local absorption and the output layer loss.
        """
        optimizer_parameters = [
            {"params": self.encoder_layer1.parameters(), "lr": 1e-4 if TRAIN_ENCODER_LAYER else 0.0},
            {"params": self.decoder_layer1.parameters(), "lr": 1e-4 if TRAIN_DECODER_LAYER else 0.0},
            {"params": self.decoder_output.parameters(), "lr": 1e-3 if TRAIN_OUTPUT_LAYER else 0.0},
        ]
        return torch.optim.Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_h2(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Extract h2 from the frozen teacher given targets x.

        This caches the top-down drive used by the student decoder. The teacher's
        decoder is evaluated to ensure internal buffers (if any) reflect h2.
        """
        _sensors, targets = batch
        latent = self.teacher.encode(targets)
        _ = self.teacher.decoder(latent)
        return self.teacher.decoder.layer2.neurons.detach()

    # -----------------------------------------------------------------------------------
    def learn(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Run the outer inference loop with local absorption and output-layer updates.

        Steps
        -----
        - Zero optimizer grads, run one inference step (prepare B h_e and predict x_hat).
        - Apply local absorption update on W_d1 (no weight transposes used for correction).
        - Backprop only through the output layer on the reconstruction loss (optional).
        - Update the reconstruction error e_0 and repeat for INFERENCE_STEPS.

        Returns
        -------
        Tensor
            The final reconstruction x_hat from the last inference step.
        """
        optimizer = self.optimizers()  # get decoder optimizers
        _sensors, targets = batch
        flattened_targets = flatten(targets, start_dim=1)
        h2 = self.sample_h2(batch)
        error = torch.zeros_like(flattened_targets)

        for _step in range(INFERENCE_STEPS):  # Inference loop with local learning
            optimizer.zero_grad(set_to_none=True)
            x_hat = self.inference_step(error, h2)  # Top-down prediction
            self.decoder_layer1.update()  # Update Wd1 locally
            nn.MSELoss(reduction="mean")(x_hat, flattened_targets.detach()).backward()  # for output layer
            error = flattened_targets - x_hat  # (B, D)
            optimizer.step()

        return x_hat

    # -----------------------------------------------------------------------------------
    def inference_step(self, error: Tensor, h2: Tensor) -> None:
        """One inference step: compute h_e from e_0 and produce a top-down prediction.

        Parameters
        ----------
        error: Tensor
            Current reconstruction error e_0.
        h2: Tensor
            Frozen top-down drive from teacher.

        Returns
        -------
        Tensor
            The reconstruction x_hat obtained after mixing the prepared correction.
        """
        correction = self.encoder_layer1(error.detach())
        self.decoder_layer1.feedback(correction)  # Prepares the correction at the decoder
        return self.forward(h2)  # Top-down prediction

    # -----------------------------------------------------------------------------------
    def forward(self, h2: Tensor) -> Tensor:
        """Top-down prediction x_hat = W_out f(W_d1 h2 + B h_e)."""
        # Top down from h2 to reconstruction x_hat and update Wd1
        h1_hat = self.decoder_layer1(h2)  # Trigger local learning in decoder layer
        return self.decoder_output(h1_hat.detach())

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Lightning training step: run learning loop and log reconstruction metrics."""
        targets = flatten(batch[1], start_dim=1)

        # Perform prediction, feedback, and learning
        x_hat = self.learn(batch)
        reconstruction_loss = nn.MSELoss(reduction="mean")(x_hat, targets)

        # Logging and metrics
        self.log("train/recon_mse", reconstruction_loss, prog_bar=True)
        self.log("train/Wd1.rms", self.decoder_layer1.weight.pow(2).mean().sqrt(), prog_bar=True)
        self.log("train/Wd1.max", self.decoder_layer1.weight.abs().max(), prog_bar=True)
        # self.log("train/Bf(h1_e).mean", self.encoder_layer1.activations.abs().mean(), prog_bar=True)

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Produce a reconstruction using the current top-down path and any queued correction."""
        h2 = self.sample_h2(batch)
        return self.forward(h2)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Predictive Coding Autoencoder (Minimal) ===")

    # Set seeds for fully deterministic behavior
    import os
    import random

    import numpy as np

    torch.manual_seed(seed=42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Removed unused augmentation imports
    from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
    from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
    from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure, ReconstructionMapParams

    # Prepare data generation
    data_param = DataParams(env_id="MiniGrid-MultiRoom-N6-v0")
    data_gen = DataGenerator(data_param)

    # Prepare data module
    datamodule_param = DataModuleParams(num_samples=320, batch_size=16, drop_last=True)
    datamodule = BaseDataModule(data_gen, datamodule_param)  # n_samples == batch_size to work

    # Prepare model and teacher
    teacher_ckpt = "models/overcomplete_autoencoders/backprop_2048.ckpt"
    ckpt = torch.load(teacher_ckpt)

    # Rebuild params from hyper_parameters if present
    if "hyper_parameters" in ckpt and "params" in ckpt["hyper_parameters"]:
        raw = ckpt["hyper_parameters"]["params"]
        # raw may be a dict or a ModelParams already
        params = raw if isinstance(raw, dict) else raw.dict()
        teacher_params = TeacherParams(**params)
    else:
        # Fallback (adjust if different in training)
        teacher_params = TeacherParams(latent_units=2048)

    teacher = Teacher(teacher_params, trainer=None)

    # Filter only weight/bias keys to ignore obsolete buffers
    state_dict = ckpt["state_dict"]
    filtered = {k: v for k, v in state_dict.items() if k.endswith(".weight") or k.endswith(".bias")}
    missing, unexpected = teacher.load_state_dict(filtered, strict=False)
    if missing:
        print("Missing keys (ignored):", missing)
    if unexpected:
        print("Unexpected keys (ignored):", unexpected)

    # Instantiate model
    model = Autoencoder(teacher)

    # Ensure datasets are prepared before directly accessing dataloader
    datamodule.setup("fit")
    first_batch = next(iter(datamodule.train_dataloader()))
    sensors0, targets0 = (t.clone() for t in first_batch)

    # Pre-training inference (CPU by default; Lightning will handle device later)
    with torch.no_grad():
        xhat0 = model.predict((sensors0, targets0))
        xhat0 = xhat0.detach()

    # Reshape reconstruction
    output_shape_tuple = tuple(model.config.output_shape)
    out_shape = (targets0.shape[0],) + output_shape_tuple
    recon0_img = xhat0.view(out_shape)

    # Figure: pre-training reconstruction
    os.makedirs("figures", exist_ok=True)
    fig_params = ReconstructionMapParams(n_samples=min(4, targets0.shape[0]), title="Pre-Training Recon")
    fig_pre = ReconstructionMapFigure(fig_params).plot(targets0.detach().cpu(), recon0_img.detach().cpu())
    fig_pre.savefig("figures/reconstruction_pre.png", dpi=120, bbox_inches="tight")
    print("Saved pre-training figure -> figures/reconstruction_pre.png")

    # Train
    trainer = pl.Trainer(max_epochs=40, enable_progress_bar=True, log_every_n_steps=1)
    trainer.fit(model, datamodule)

    # Post-training reconstruction using same batch
    with torch.no_grad():
        xhat1 = model.predict((sensors0, targets0))
        xhat1 = xhat1.detach()
    recon1_img = xhat1.view(out_shape)

    fig_params_post = ReconstructionMapParams(n_samples=min(4, targets0.shape[0]), title="Post-Training Recon")
    fig_post = ReconstructionMapFigure(fig_params_post).plot(targets0.detach().cpu(), recon1_img.detach().cpu())
    fig_post.savefig("figures/reconstruction_post.png", dpi=120, bbox_inches="tight")
    print("Saved post-training figure -> figures/reconstruction_post.png")
    print("=== Done ===")
