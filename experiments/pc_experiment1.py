import math
from typing import Literal, Optional, Tuple

import torch
from lightning import pytorch as pl
from pydantic import BaseModel, Field
from torch import Tensor, flatten, nn
from torch.optim import Optimizer

from ehc_sn.models.ann.sparse_autoencoder import Autoencoder as Teacher
from ehc_sn.models.ann.sparse_autoencoder import ModelParams as TeacherParams

# -------------------------------------------------------------------------------------------
FEEDBACK_MODE: Literal["random", "identity"] = "random"
COMBINATION_MODE: Literal["identity", "random"] = "identity"
ACTIVATION_FN: bool = True


# -------------------------------------------------------------------------------------------
class EncoderLayer(nn.Linear):
    """Error-to-hidden mapping with local feedback projection matrix F."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, **kwargs) -> None:
        super().__init__(in_features, out_features, bias=False, **kwargs)
        self.register_buffer("F", torch.empty(out_features, in_features))  # feedback projection
        self.activation = nn.GELU() if ACTIVATION_FN else nn.Identity()
        self.register_buffer("currents", None)
        self.register_buffer("activations", None)
        self.init_feedback()  # default init

    def forward(self, input: Tensor) -> Tensor:
        self.currents = super().forward(input.detach())
        self.activations = self.activation(self.currents)
        return self.activations

    def feedback(self, error: Tensor) -> None:
        delta = error.detach() @ self.F.T  # (B, H1)
        local_loss = (self.activations * delta).sum() / error.shape[0]
        local_loss.backward()

    def init_feedback(self) -> None:
        """Initialize feedback matrix F based on selected mode."""
        if FEEDBACK_MODE == "identity":
            self.F.copy_(torch.eye(*self.F.shape, device=self.F.device, dtype=self.F.dtype))
        elif FEEDBACK_MODE == "random":
            torch.nn.init.kaiming_uniform_(self.F, a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown feedback mode: {FEEDBACK_MODE}")

    @property
    def signal_mean(self) -> Optional[float]:
        if self.activations is not None:
            return self.activations.abs().mean().item()
        return None


# -------------------------------------------------------------------------------------------
class DecoderLayer(nn.Linear):
    """Frozen first decoder layer W_{d1} mapping h2 -> h1 base with combination matrix B."""

    def __init__(self, in_features: int, out_features: int, **kwargs) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.register_buffer("B", torch.empty(out_features, out_features))  # combination / modulation
        self.activation = nn.GELU() if ACTIVATION_FN else nn.Identity()
        self.register_buffer("currents", None)
        self.register_buffer("activations", None)
        self.init_combination()  # default init

    def forward(self, inputs: Tensor, feedback: Tensor) -> Tensor:
        self.currents = super().forward(inputs.detach())
        self.currents += feedback.detach() @ self.B.T
        self.activations = self.activation(self.currents)
        return self.activations

    def feedback(self, error: Tensor) -> Tensor:
        """Compute combination projection B h_e (no gradient)."""
        raise RuntimeError("This experiment assumes fixed decoder parameters.")

    def init_combination(self) -> None:
        if COMBINATION_MODE == "identity":
            self.B.copy_(torch.eye(self.B.shape[0], device=self.B.device, dtype=self.B.dtype))
        elif COMBINATION_MODE == "random":
            torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        else:
            raise ValueError(f"Unknown combination mode: {COMBINATION_MODE}")


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    """Predictive-coding style experiment with explicit inference + local learning split.

    Only the error-to-hidden weights (encoder_layer1) are updated via a surrogate
    local gradient using a feedback / alignment matrix F. Decoder layers and
    teacher remain frozen.
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
        self.decoder_layer1 = DecoderLayer(n_h2, n_h1, bias=True)
        self.encoder_layer1 = EncoderLayer(n_sensors, n_h1, bias=False)  # ensure no constant drive

    # -----------------------------------------------------------------------------------
    def configure_optimizers(self) -> Optimizer:
        l1_encoder = {"params": self.encoder_layer1.parameters(), "lr": 1e-3}
        l0_decoder = {"params": self.teacher.decoder.output.parameters(), "lr": 0.0}
        return torch.optim.Adam([l1_encoder, l0_decoder])

    # -----------------------------------------------------------------------------------
    @torch.no_grad()
    def sample_h2(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        _sensors, targets = batch
        latent = self.teacher.encode(targets)
        _ = self.teacher.decoder(latent)
        return self.teacher.decoder.layer2.neurons.detach()

    # -----------------------------------------------------------------------------------
    def forward(self, batch: Tuple[Tensor, Tensor], h2: Tensor) -> Tuple[Tensor, Tensor]:
        sensors, targets = batch
        flat_sensors = flatten(sensors, start_dim=1)
        zeros = torch.zeros(h2.shape[0], self.encoder_layer1.out_features, device=h2.device)
        with torch.no_grad():
            h1_base = self.decoder_layer1(h2, zeros)  # (B, H1)
            x_base = self.teacher.decoder.output(h1_base)
        error = flat_sensors - x_base
        h1_err = self.encoder_layer1(error)
        h1_hat = self.decoder_layer1(h2, h1_err)
        x_hat = self.teacher.decoder.output(h1_hat.detach())
        return error, x_hat

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        targets = flatten(batch[1], start_dim=1)
        h2 = self.sample_h2(batch)
        optimizer = self.optimizers()

        # Run the first forward pass to get error and prediction
        error, prediction = self(batch, h2)

        # Update encoder layer with local feedback
        optimizer.zero_grad()
        self.encoder_layer1.feedback(error)
        reconstruction_loss = nn.MSELoss(reduce="mean")(prediction, targets.detach())
        self.manual_backward(reconstruction_loss)
        optimizer.step()

        # Log metrics to monitor progress
        self.log("train/recon_mse", reconstruction_loss, prog_bar=True)
        self.log("train/Bf(h_e).mean", self.encoder_layer1.signal_mean, prog_bar=True)


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
    first_batch = next(iter(datamodule.train_dataloader()))
    sensors0, targets0 = (t.clone() for t in first_batch)

    # Pre-training inference (CPU by default; Lightning will handle device later)
    with torch.no_grad():
        h2_0 = model.sample_h2((sensors0, targets0))
    _e0_prev0, xhat0 = model((sensors0, targets0), h2_0)
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
    trainer = pl.Trainer(max_epochs=40, enable_progress_bar=True)
    trainer.fit(model, datamodule)

    # Post-training reconstruction using same batch & latent
    with torch.no_grad():
        _e0_prev1, xhat1 = model((sensors0, targets0), h2_0)
        xhat1 = xhat1.detach()
    recon1_img = xhat1.view(out_shape)

    fig_params_post = ReconstructionMapParams(n_samples=min(4, targets0.shape[0]), title="Post-Training Recon")
    fig_post = ReconstructionMapFigure(fig_params_post).plot(targets0.detach().cpu(), recon1_img.detach().cpu())
    fig_post.savefig("figures/reconstruction_post.png", dpi=120, bbox_inches="tight")
    print("Saved post-training figure -> figures/reconstruction_post.png")
    print("=== Done ===")
