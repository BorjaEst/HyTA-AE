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
    """Linear layer mapping flattened reconstruction error to hidden corrective code.

    Holds last pre-activation (currents) for local gradient computation.
    """

    def __init__(self, *args, device=None, dtype=None, **kwargs) -> None:
        super().__init__(*args, device=device, dtype=dtype, **kwargs)
        self.activation = nn.GELU()
        self.register_buffer("currents", None)
        self.register_buffer("activations", None)

    def forward(self, input: Tensor) -> Tensor:
        # Local mapping ONLY; detach upstream to forbid global credit.
        input_local = input.detach()
        self.currents = super().forward(input_local)
        self.activations = self.activation(self.currents)
        return self.activations


# -------------------------------------------------------------------------------------------
class DecoderLayer(nn.Linear):
    """Frozen first decoder layer W_{d1} mapping h2 -> h1 base."""

    def forward(self, inputs: Tensor) -> Tensor:
        return super().forward(inputs.detach())


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

        # Feedback / combination matrices (buffers)
        self.register_buffer("F", torch.empty(n_h1, n_sensors))  # feedback matrix
        self.register_buffer("B", torch.empty(n_h1, n_h1))  # combination matrix
        self._init_feedback_matrices()

        # Experiment state / caches
        self.cached_batch: Optional[Tuple[Tensor, Tensor]] = None
        self.stored_h2: Optional[Tensor] = None
        self.prev_error_norm: Optional[Tensor] = None

    # -----------------------------------------------------------------------------------
    def _init_feedback_matrices(self) -> None:
        device = self.decoder_layer1.weight.device
        n_h1, n_sensors = self.F.shape
        mode = self.pc_params.feedback_mode
        if mode == "symmetric":
            W_out = self.teacher.decoder.output
            self.F.copy_(W_out.weight.T.detach())
        elif mode == "identity":
            # Assume dimensions match if user requests identity
            self.F.copy_(torch.eye(n_h1, n_sensors))
        else:  # random
            torch.nn.init.kaiming_uniform_(self.F, a=math.sqrt(5))

        if self.pc_params.combination_mode == "identity":
            self.B.copy_(torch.eye(n_h1))
        else:
            torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self.F = self.F.to(device)
        self.B = self.B.to(device)

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        # Optimizer kept only for logging compatibility; manual update bypasses .step()
        # (Return a dummy optimizer on encoder parameters.)
        return torch.optim.SGD(self.encoder_layer1.parameters(), lr=self.pc_params.learning_rate)

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
        flat_sensors = flatten(sensors, start_dim=1)

        with torch.no_grad():
            h1_base = self.decoder_layer1(h2)  # (B, H1)
            x_base = self.teacher.decoder.output(h1_base)  # (B, D)
        e0_prev = flat_sensors - x_base  # (B, D)

        # Corrective code (builds local graph for W_e only)
        h_e = self.encoder_layer1(e0_prev)
        h1_hat = h1_base + (h_e @ self.B.T)
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
        self.prev_error_norm = None

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:  # type: ignore[override]
        batch = self.cached_batch  # type: ignore[assignment]
        h2 = self.stored_h2

        out = self.inference(batch, h2)
        e0_prev = out["e0_prev"]
        e0 = out["e0"]
        h_e = out["h_e"]
        # Strictly local surrogate (Copilot-Processing Section 6.2):
        # delta_tilde = F' e0_prev ; grad_W = ((delta_tilde ⊙ f'(u)) e0_prev^T)/B
        with torch.no_grad():
            delta_tilde = e0_prev @ self.F.T  # (B, H1)

        pre_act = self.encoder_layer1.currents  # u = W_e e0_prev
        gelu_prime = gelu_derivative(pre_act)
        surrogate_term = delta_tilde * gelu_prime  # (B, H1)

        batch_size = e0_prev.shape[0]
        grad_W = surrogate_term.T @ e0_prev / batch_size
        grad_b = surrogate_term.mean(dim=0)

        # (No sparsity regularization or weight decay in simplified version)

        lr = self.pc_params.learning_rate
        with torch.no_grad():
            self.encoder_layer1.weight -= lr * grad_W
            if self.encoder_layer1.bias is not None:
                self.encoder_layer1.bias -= lr * grad_b

        # Metrics
        recon_mse = e0.pow(2).mean()
        correction_norm = h_e.norm(p=2, dim=1).mean()
        if self.prev_error_norm is None:
            stability = torch.tensor(0.0, device=self.device)
        else:
            stability = (recon_mse - self.prev_error_norm).abs()
        self.prev_error_norm = recon_mse.detach()

        self.log("train/recon_mse", recon_mse, prog_bar=True)
        self.log("train/correction_norm", correction_norm, prog_bar=False)
        self.log("train/error_stability", stability, prog_bar=False)
        self.log("train/weight_grad_norm", grad_W.norm(p=2), prog_bar=False)

        return recon_mse


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Test predictive coding first model implementation
    print("=== Testing Predictive Coding Autoencoder Model ===")

    # Set seeds for fully deterministic behavior
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

    # Instantiate and train themodel
    model = Autoencoder(teacher)
    pl.Trainer(max_epochs=200, deterministic=True, enable_progress_bar=True).fit(model, datamodule)
    print("=== Test Completed ===")
