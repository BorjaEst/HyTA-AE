from typing import Optional

from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F


# -------------------------------------------------------------------------------------------
class MetricsLogger:
    """Comprehensive metrics logger for autoencoder experiments.

    Logs reconstruction quality, sparsity, activation statistics, and
    biological plausibility metrics for entorhinal-hippocampal circuit models.
    """

    def __init__(self, model: LightningModule, sparsity_threshold: float = 0.01):
        """Initialize metrics logger.

        Args:
            model: LightningModule instance to log metrics to
            sparsity_threshold: Threshold for determining if a latent unit is inactive
        """
        self.model = model
        self.sparsity_threshold = sparsity_threshold

    # -----------------------------------------------------------------------------------
    def log_reconstruction_bce(
        self, reconstruction: Tensor, targets: Tensor, prefix: str = "train", prog_bar: bool = True
    ) -> Tensor:
        """Log binary cross-entropy reconstruction loss.

        Args:
            reconstruction: Model output (probabilities after sigmoid)
            targets: Ground truth binary maps
            prefix: Metric prefix (e.g., 'train', 'val')
            prog_bar: Whether to show in progress bar

        Returns:
            BCE loss value
        """
        loss = F.binary_cross_entropy(reconstruction, targets, reduction="mean")
        on_step = prefix == "train"
        on_epoch = prefix != "train"
        self.model.log(f"{prefix}/reconstruction_bce", loss, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch)
        return loss

    # -----------------------------------------------------------------------------------
    def log_reconstruction_mse(
        self, reconstruction: Tensor, targets: Tensor, prefix: str = "val", prog_bar: bool = False
    ) -> Tensor:
        """Log mean squared error reconstruction loss.

        Args:
            reconstruction: Model output
            targets: Ground truth maps
            prefix: Metric prefix (e.g., 'train', 'val')
            prog_bar: Whether to show in progress bar

        Returns:
            MSE loss value
        """
        loss = F.mse_loss(reconstruction, targets, reduction="mean")
        on_step = prefix == "train"
        on_epoch = prefix != "train"
        self.model.log(f"{prefix}/reconstruction_mse", loss, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch)
        return loss

    # -----------------------------------------------------------------------------------
    def log_sparsity(self, latent: Tensor, prefix: str = "train", prog_bar: bool = True) -> Tensor:
        """Log sparsity rate of latent activations.

        Args:
            latent: Latent code tensor
            prefix: Metric prefix (e.g., 'train', 'val')
            prog_bar: Whether to show in progress bar

        Returns:
            Sparsity rate (fraction of inactive units)
        """
        sparsity_rate = (latent.abs() < self.sparsity_threshold).float().mean()
        on_step = prefix == "train"
        on_epoch = prefix != "train"
        self.model.log(f"{prefix}/sparsity_rate", sparsity_rate, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch)
        return sparsity_rate

    # -----------------------------------------------------------------------------------
    def log_activation_stats(self, latent: Tensor, prefix: str = "val") -> None:
        """Log activation statistics for latent representations.

        Args:
            latent: Latent code tensor
            prefix: Metric prefix (e.g., 'train', 'val')
        """
        on_step = prefix == "train"
        on_epoch = prefix != "train"

        # Mean activation level
        mean_activation = latent.mean()
        self.model.log(f"{prefix}/latent_mean", mean_activation, prog_bar=False, on_step=on_step, on_epoch=on_epoch)

        # Standard deviation
        std_activation = latent.std()
        self.model.log(f"{prefix}/latent_std", std_activation, prog_bar=False, on_step=on_step, on_epoch=on_epoch)

        # L1 norm (sparsity measure)
        l1_norm = latent.abs().mean()
        self.model.log(f"{prefix}/latent_l1", l1_norm, prog_bar=False, on_step=on_step, on_epoch=on_epoch)

        # Active units per sample
        active_per_sample = (latent.abs() > self.sparsity_threshold).float().sum(dim=1).mean()
        self.model.log(f"{prefix}/active_units", active_per_sample, prog_bar=False, on_step=on_step, on_epoch=on_epoch)

    # -----------------------------------------------------------------------------------
    def log_layer_alignment(
        self, decoder_hidden: Tensor, encoder_hidden: Tensor, layer_idx: int, prefix: str = "val"
    ) -> Tensor:
        """Log alignment loss between decoder and encoder hidden layers.

        Args:
            decoder_hidden: Decoder layer activations
            encoder_hidden: Encoder layer activations (detached)
            layer_idx: Layer index for naming
            prefix: Metric prefix (e.g., 'train', 'val')

        Returns:
            MSE alignment loss
        """
        loss = F.mse_loss(decoder_hidden, encoder_hidden.detach(), reduction="mean")
        on_step = prefix == "train"
        on_epoch = prefix != "train"
        self.model.log(f"{prefix}/h{layer_idx}_alignment", loss, prog_bar=False, on_step=on_step, on_epoch=on_epoch)
        return loss

    # -----------------------------------------------------------------------------------
    def log_loss_component(
        self, loss_value: Tensor, loss_name: str, prefix: str = "train", prog_bar: bool = False
    ) -> None:
        """Log a generic loss component.

        Args:
            loss_value: Computed loss value
            loss_name: Name of the loss component (e.g., 'gramian', 'homeostatic')
            prefix: Metric prefix (e.g., 'train', 'val')
            prog_bar: Whether to show in progress bar
        """
        on_step = prefix == "train"
        on_epoch = prefix != "train"
        self.model.log(f"{prefix}/{loss_name}", loss_value, prog_bar=prog_bar, on_step=on_step, on_epoch=on_epoch)

    # -----------------------------------------------------------------------------------
    def log_all_training(
        self, reconstruction: Tensor, latent: Tensor, targets: Tensor, include_stats: bool = False
    ) -> None:
        """Log all common training metrics at once.

        Args:
            reconstruction: Model output
            latent: Latent code tensor
            targets: Ground truth maps
            include_stats: Whether to include activation statistics (adds overhead)
        """
        self.log_reconstruction_bce(reconstruction, targets, prefix="train", prog_bar=True)
        self.log_sparsity(latent, prefix="train", prog_bar=True)
        if include_stats:
            self.log_activation_stats(latent, prefix="train")

    # -----------------------------------------------------------------------------------
    def log_all_validation(
        self, reconstruction: Tensor, latent: Tensor, targets: Tensor, include_stats: bool = True
    ) -> None:
        """Log all common validation metrics at once.

        Args:
            reconstruction: Model output
            latent: Latent code tensor
            targets: Ground truth maps
            include_stats: Whether to include activation statistics
        """
        self.log_reconstruction_bce(reconstruction, targets, prefix="val", prog_bar=True)
        self.log_reconstruction_mse(reconstruction, targets, prefix="val", prog_bar=False)
        self.log_sparsity(latent, prefix="val", prog_bar=True)
        if include_stats:
            self.log_activation_stats(latent, prefix="val")
