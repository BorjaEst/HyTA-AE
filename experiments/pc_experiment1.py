import math
from typing import Dict, Literal, Optional, Tuple

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field
from torch import Tensor, flatten, nn
from torch.optim import Optimizer

from ehc_sn.models.ann.sparse_autoencoder import Autoencoder as Teacher
from ehc_sn.models.ann.sparse_autoencoder import ModelParams as TeacherParams


# -------------------------------------------------------------------------------------------
class PCExperimentParams(BaseModel):
    """Minimal configuration for the predictive coding experiment (reconstruction only)."""

    learning_rate: float = Field(1e-3, ge=0.0)
    feedback_mode: Literal["symmetric", "random", "identity"] = "random"
    combination_mode: Literal["identity", "random"] = "identity"


def gelu_derivative(x: Tensor) -> Tensor:
    """Analytic derivative of GELU (approx standard formulation).

    GELU(x) = 0.5 x [1 + erf(x / sqrt(2))]
    d/dx GELU(x) = 0.5 * (1 + erf(x / sqrt(2))) + (x * exp(-x^2 /2)) / sqrt(2*pi)
    """
    import torch

    sqrt_2 = math.sqrt(2.0)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    erf_term = torch.erf(x / sqrt_2)
    exp_term = torch.exp(-0.5 * x * x)
    return 0.5 * (1.0 + erf_term) + (x * exp_term) / sqrt_2pi


# -------------------------------------------------------------------------------------------
class EncoderLayer(nn.Linear):
    """Error-to-hidden mapping with local feedback projection matrix F.

    F projects input-layer reconstruction error (dimension D) into hidden corrective
    units (H1) providing a local teaching signal (delta_tilde = F e0_prev).
    """

    def __init__(self, in_features: int, out_features: int, *, device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, device=device, dtype=dtype, **kwargs)
        self.activation = nn.GELU()
        self.register_buffer("currents", None)
        self.register_buffer("activations", None)
        self.register_buffer("F", torch.empty(out_features, in_features))  # feedback projection

    def init_feedback(self, mode: Literal["symmetric", "random", "identity"], teacher_output_layer: nn.Module) -> None:
        """Initialize feedback matrix F based on selected mode."""
        if mode == "symmetric":
            W_out = teacher_output_layer
            self.F.copy_(W_out.weight.T.detach())  # shape (H1,D)
        elif mode == "identity":
            # assume shapes compatible (may be rectangular -> eye crops/pads not handled here)
            rows, cols = self.F.shape
            eye = torch.eye(rows, cols, device=self.F.device, dtype=self.F.dtype)
            self.F.copy_(eye)
        else:  # random
            torch.nn.init.kaiming_uniform_(self.F, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        input_local = input.detach()
        self.currents = super().forward(input_local)
        self.activations = self.activation(self.currents)
        return self.activations

    def feedback(self, error: Tensor) -> Tensor:
        """Project reconstruction error locally and backprop through this layer only.

        Parameters
        ----------
        error : Tensor
            Reconstruction error at sensory layer (B, D) BEFORE correction (e0_prev).

        Returns
        -------
        Tensor
            Local projected signal delta_tilde = F e (B, H1) used as teaching signal.

        Notes
        -----
        We build a purely local scalar objective L_local = sum_b,h a_{b,h} * delta_tilde_{b,h}
        so that autograd yields: dL/dW = (delta_tilde ⊙ f'(u)) input^T, matching the
        hand-crafted surrogate gradient. Inputs are detached so no upstream credit flows.
        Batch-mean normalization keeps update scale comparable to earlier manual rule.
        """
        if self.activations is None or self.currents is None:
            raise RuntimeError("Forward pass must be executed before feedback().")
        # Local projection (no gradient path through error or F)
        delta_tilde = error.detach() @ self.F.T  # (B, H1)
        batch_size = error.shape[0]
        # Local scalar objective whose gradient matches desired update direction
        local_loss = (self.activations * delta_tilde).sum() / batch_size
        # Zero existing grads (in-case of accumulation) then backward through this layer only
        if self.weight.grad is not None:
            self.weight.grad.zero_()
        if self.bias is not None and self.bias.grad is not None:
            self.bias.grad.zero_()
        local_loss.backward()
        return delta_tilde


# -------------------------------------------------------------------------------------------
class DecoderLayer(nn.Linear):
    """Frozen first decoder layer W_{d1} mapping h2 -> h1 base with combination matrix B."""

    def __init__(self, in_features: int, out_features: int, *, device=None, dtype=None, **kwargs) -> None:
        super().__init__(in_features, out_features, device=device, dtype=dtype, **kwargs)
        self.register_buffer("B", torch.empty(out_features, out_features))  # combination / modulation

    def init_combination(self, mode: Literal["identity", "random"]) -> None:
        if mode == "identity":
            self.B.copy_(torch.eye(self.B.shape[0], device=self.B.device, dtype=self.B.dtype))
        else:
            torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        return super().forward(inputs.detach())

    def feedback(self, error: Tensor) -> Tensor:
        """Compute combination projection B h_e (no gradient)."""
        raise RuntimeError("This experiment assumes fixed decoder parameters.")


# -------------------------------------------------------------------------------------------
class ErrorLayer(nn.Module):
    """Computes reconstruction error e0 = x - x_hat (flattened)."""

    def forward(self, sensors: Tensor, predictions: Tensor) -> Tensor:
        flat_sensors = flatten(sensors, start_dim=1)
        return flat_sensors - predictions  # e0


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    """Predictive-coding style experiment with explicit inference + local learning split.

    Only the error-to-hidden weights (encoder_layer1) are updated via a surrogate
    local gradient using a feedback / alignment matrix F. Decoder layers and
    teacher remain frozen.
    """

    def __init__(self, teacher: nn.Module, pc_params: Optional[PCExperimentParams] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.config = params = teacher.config  # teacher's (frozen) params
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        self.pc_params = pc_params or PCExperimentParams()
        self.automatic_optimization = False  # manual local update

        # Dimensions
        self.n_hidden = n_h1, n_h2 = params.layer1_units, params.layer2_units
        self.n_sensors = n_sensors = math.prod(params.output_shape)

        # Modules
        self.teacher = teacher
        self.decoder_layer1 = DecoderLayer(n_h2, n_h1, bias=True)  # W_{d1}
        self.encoder_layer1 = EncoderLayer(n_sensors, n_h1, bias=True)  # W_e
        self.error_layer = ErrorLayer()

        # Reconstruction energy (no sparsity / weight decay in simplified version)
        self.reconstruction_loss = nn.MSELoss(reduction="mean")

        # Initialize local feedback/combination matrices inside layers
        self.encoder_layer1.init_feedback(self.pc_params.feedback_mode, self.teacher.decoder.output)
        self.decoder_layer1.init_combination(self.pc_params.combination_mode)

        # Experiment state / caches
        self.cached_batch: Optional[Tuple[Tensor, Tensor]] = None
        self.stored_h2: Optional[Tensor] = None

    # -----------------------------------------------------------------------------------
    def _init_feedback_matrices(self) -> None:  # kept for backward compatibility (noop)
        pass

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        return torch.optim.SGD(
            self.encoder_layer1.parameters(),
            lr=self.pc_params.learning_rate,
        )

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_h2(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        _sensors, targets = batch
        latent = self.teacher.encode(targets)
        _ = self.teacher.decoder(latent)
        return self.teacher.decoder.layer2.neurons.detach()

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, batch: Tuple[Tensor, Tensor], h2: Tensor) -> Tensor:  # type: ignore[override]
        """Lightning forward: returns final reconstruction only (for compatibility)."""
        out = self.inference(batch, h2)
        return out["x_hat"]

    # -----------------------------------------------------------------------------------
    def inference(self, batch: Tuple[Tensor, Tensor], h2: Tensor) -> Dict[str, Tensor]:
        sensors, targets = batch
        device = self.encoder_layer1.weight.device
        sensors = sensors.to(device)
        targets = targets.to(device)
        h2 = h2.to(device)
        flat_sensors = flatten(sensors, start_dim=1)

        with torch.no_grad():
            h1_base = self.decoder_layer1(h2)  # (B, H1)
            x_base = self.teacher.decoder.output(h1_base)  # (B, D)
        e0_prev = flat_sensors - x_base  # (B, D)

        # Corrective code (builds local graph for W_e only)
        h_e = self.encoder_layer1(e0_prev)
        h1_hat = h1_base + (h_e @ self.decoder_layer1.B.T)
        with torch.no_grad():
            x_hat = self.teacher.decoder.output(h1_hat)  # (B, D)
        e0 = flat_sensors - x_hat

        return {
            "h1_base": h1_base,
            "x_base": x_base,
            "e0_prev": e0_prev,
            "h_e": h_e,
            "h1_hat": h1_hat,
            "x_hat": x_hat,
            "e0": e0,
        }

    # -----------------------------------------------------------------------------------
    def on_fit_start(self) -> None:
        dl = self.trainer.datamodule.train_dataloader()
        raw_batch = next(iter(dl))
        batch = tuple(t.to(self.device) for t in raw_batch)  # assume tuple
        self.cached_batch = batch  # type: ignore[assignment]
        self.stored_h2 = self.sample_h2(batch)  # type: ignore[arg-type]

    # no stability tracking needed when only logging reconstruction

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        batch = self.cached_batch  # type: ignore[assignment]
        h2 = self.stored_h2

        out = self.inference(batch, h2)
        e0_prev = out["e0_prev"]
        e0 = out["e0"]
        # Local autograd-based surrogate update using optimizer
        optimizer = self.optimizers()  # retrieve configured optimizer (SGD on encoder_layer1)
        optimizer.zero_grad()
        _ = self.encoder_layer1.feedback(e0_prev)  # computes local backward and populates grads
        optimizer.step()

        # Metric (only reconstruction MSE)
        recon_mse = e0.pow(2).mean()
        self.log("train/recon_mse", recon_mse, prog_bar=True)

        return recon_mse


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test predictive coding first model implementation
    print("=== Testing Predictive Coding Autoencoder Model ===")

    # Set seeds for fully deterministic behavior
    import os
    import random

    import numpy as np

    torch.manual_seed(seed=42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from ehc_sn.augmentation.incomplete_maps import Augmentation, ComposeParams
    from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
    from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
    from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure, ReconstructionMapParams

    # Prepare data generation
    data_param = DataParams(env_id="MiniGrid-MultiRoom-N6-v0")
    data_gen = DataGenerator(data_param)

    # Prepare data module
    datamodule_param = DataModuleParams(num_samples=32, batch_size=32, drop_last=False)
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
    # Prepare a batch for before/after comparison (reuse first train batch)
    first_batch = next(iter(datamodule.train_dataloader()))
    sensors0, targets0 = (t.clone() for t in first_batch)  # clone to avoid in-place side effects

    # Move model & batch to device early
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sensors0 = sensors0.to(device)
    targets0 = targets0.to(device)
    with torch.no_grad():
        h2_0 = model.sample_h2((sensors0, targets0)).to(device)
        out0 = model.inference((sensors0, targets0), h2_0)
        xhat0 = out0["x_hat"].detach()

    # Reshape flattened reconstruction to spatial map
    # output_shape stored as list -> convert to tuple for view; infer channel handling
    output_shape_tuple = tuple(model.config.output_shape)
    out_shape = (targets0.shape[0],) + output_shape_tuple
    try:
        recon0_img = xhat0.view(out_shape)
    except RuntimeError:
        # Attempt to treat as (N, H, W) if channel mismatch
        if len(output_shape_tuple) == 2:
            recon0_img = xhat0.view(targets0.shape[0], *output_shape_tuple)
        else:
            recon0_img = targets0.clone()

    # Figure: pre-training reconstruction
    os.makedirs("figures", exist_ok=True)
    fig_params = ReconstructionMapParams(n_samples=min(4, targets0.shape[0]), title="Pre-Training Recon")
    fig_pre = ReconstructionMapFigure(fig_params).plot(targets0.detach().cpu(), recon0_img.detach().cpu())
    fig_pre.savefig("figures/reconstruction_pre.png", dpi=120, bbox_inches="tight")
    print("Saved pre-training reconstruction figure to figures/reconstruction_pre.png")

    # Train
    trainer = pl.Trainer(max_epochs=200, enable_progress_bar=True)
    trainer.fit(model, datamodule)

    # Post-training reconstruction using same batch & latent
    with torch.no_grad():
        h2_0 = h2_0.to(model.device)
        out1 = model.inference((sensors0, targets0), h2_0)
        xhat1 = out1["x_hat"].detach()
    try:
        recon1_img = xhat1.view(out_shape)
    except RuntimeError:
        if len(output_shape_tuple) == 2:
            recon1_img = xhat1.view(targets0.shape[0], *output_shape_tuple)
        else:
            recon1_img = targets0.clone()

    fig_params_post = ReconstructionMapParams(n_samples=min(4, targets0.shape[0]), title="Post-Training Recon")
    fig_post = ReconstructionMapFigure(fig_params_post).plot(targets0.detach().cpu(), recon1_img.detach().cpu())
    fig_post.savefig("figures/reconstruction_post.png", dpi=120, bbox_inches="tight")
    print("Saved post-training reconstruction figure to figures/reconstruction_post.png")

    print("=== Test Completed ===")
