from typing import List, Optional, Tuple

import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F

# -------------------------------------------------------------------------------------------
# MetricsLogger Class
# -------------------------------------------------------------------------------------------


class MetricsLogger:
    """
    Unified metrics logger for EHC autoencoder experiments.

    Logs all metrics defined in the paper:
    - Reconstruction: MSE, SSIM (optional for now)
    - Completion (FCMT): hidden-region MSE
    - Pattern separation (DG): correlation off-diagonal
    - DG sparsity: Hoyer S+, active rate ρ_ε
    - CA3 sparsity: Hoyer S+, active rate ρ_ε
    - Encoder-decoder alignment: CKA (per layer and mean)
    """

    def __init__(self, lightning_module: LightningModule, eps_sparsity: float = 0.02):
        """
        Initialize metrics logger.

        Args:
            lightning_module: Parent Lightning module for self.log access
            eps_sparsity: Threshold for sparsity_rate computation
        """
        self.module = lightning_module
        self.eps_sparsity = eps_sparsity

    # -----------------------------------------------------------------------------------
    def log_training(self, batch: Tuple[Tensor, Tensor], output: Tuple[Tensor, Tensor, Tensor]) -> None:
        """
        Log training metrics (lightweight, avoid expensive computations).

        Args:
            batch: (sensors, targets) where sensors include mask channel
            output: (reconstruction, patterns, state) from model forward
        """
        reconstruction, patterns, state = output
        sensors, targets = batch

        # Reconstruction MSE (full image)
        mse_full = F.mse_loss(reconstruction, targets)
        self.module.log("train/mse_reconstruction", mse_full, on_step=True, on_epoch=True, prog_bar=True)

        # DG sparsity (on patterns)
        m_dg = patterns.abs()
        hoyer_dg = hoyer_plus(m_dg, dim=-1).mean()
        rate_dg = sparsity_rate(m_dg, eps=self.eps_sparsity, dim=-1).mean()

        self.module.log("train/patterns_hoyer", hoyer_dg, on_step=False, on_epoch=True)
        self.module.log("train/patterns_active_rate", rate_dg, on_step=False, on_epoch=True)

        # CA3 sparsity (on state/latent_post)
        m_ca3 = state.abs()
        hoyer_ca3 = hoyer_plus(m_ca3, dim=-1).mean()
        rate_ca3 = sparsity_rate(m_ca3, eps=self.eps_sparsity, dim=-1).mean()

        self.module.log("train/latent_post_hoyer", hoyer_ca3, on_step=False, on_epoch=True)
        self.module.log("train/latent_post_active_rate", rate_ca3, on_step=False, on_epoch=True)

    # -----------------------------------------------------------------------------------
    def log_validation(self, batch: Tuple[Tensor, Tensor], all_signals: List[Tensor]) -> None:
        """
        Log validation metrics (comprehensive).

        Args:
            batch: (sensors, targets) where sensors include mask channel
            all_signals: [h1_enc, latent_pre, patterns, latent_post, h1_dec, reconstruction]
        """
        sensors, targets = batch

        # Unpack signals
        h1_enc, latent_pre, patterns, latent_post, h1_dec, reconstruction = all_signals

        # Extract mask: sensors[:, 1] is the visibility mask (1 = visible, 0 = hidden)
        mask = sensors[:, 1:2] if sensors.shape[1] > 1 else torch.ones_like(targets)

        # -----------------------------------------------------------------------------------
        # Reconstruction metrics
        # -----------------------------------------------------------------------------------

        # Full-image MSE (primary)
        mse_full = F.mse_loss(reconstruction, targets)
        self.module.log("val/mse_reconstruction", mse_full, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Hidden-region MSE (FCMT completion metric)
        mse_masked = masked_mse(reconstruction, targets, 1 - mask)
        self.module.log("val/mse_masked", mse_masked, on_step=False, on_epoch=True, sync_dist=True)

        # -----------------------------------------------------------------------------------
        # Pattern separation (DG)
        # -----------------------------------------------------------------------------------

        # Correlation off-diagonal (decorrelation metric)
        corr_offdiag_dg = corr_offdiag(patterns)
        self.module.log("val/patterns_corr_offdiag", corr_offdiag_dg, on_step=False, on_epoch=True, sync_dist=True)

        # DG sparsity
        m_dg = patterns.abs()
        hoyer_dg = hoyer_plus(m_dg, dim=-1).mean()
        rate_dg = sparsity_rate(m_dg, eps=self.eps_sparsity, dim=-1).mean()

        self.module.log("val/patterns_hoyer", hoyer_dg, on_step=False, on_epoch=True, sync_dist=True)
        self.module.log("val/patterns_active_rate", rate_dg, on_step=False, on_epoch=True, sync_dist=True)

        # -----------------------------------------------------------------------------------
        # CA3 latent sparsity
        # -----------------------------------------------------------------------------------

        m_ca3 = latent_post.abs()
        hoyer_ca3 = hoyer_plus(m_ca3, dim=-1).mean()
        rate_ca3 = sparsity_rate(m_ca3, eps=self.eps_sparsity, dim=-1).mean()

        self.module.log("val/latent_post_hoyer", hoyer_ca3, on_step=False, on_epoch=True, sync_dist=True)
        self.module.log("val/latent_post_active_rate", rate_ca3, on_step=False, on_epoch=True, sync_dist=True)

        # -----------------------------------------------------------------------------------
        # Encoder-decoder alignment (CKA)
        # -----------------------------------------------------------------------------------

        # Compare paired encoder/decoder hidden layers at same depth
        # h2_enc (pre-latent encoder) <-> h2_dec (post-latent decoder)
        cka_h2 = linear_cka(latent_pre, latent_post)
        self.module.log("val/align_cka_latent", cka_h2, on_step=False, on_epoch=True, sync_dist=True)

        # Compare paired encoder/decoder hidden layers at same depth
        # h1_enc (first encoder layer) <-> h1_dec (first decoder layer)
        cka_h1 = linear_cka(h1_enc, h1_dec)
        self.module.log("val/align_cka_hidden", cka_h1, on_step=False, on_epoch=True, sync_dist=True)


# -------------------------------------------------------------------------------------------
# Core Metric Functions
# -------------------------------------------------------------------------------------------


def hoyer_plus(m: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    """
    Compute Hoyer sparsity on nonnegative magnitude vector.

    S_Hoyer^+(m) = (sqrt(n) - ||m||_1 / ||m||_2) / (sqrt(n) - 1) ∈ [0,1]

    where 0 indicates dense/uniform activity and 1 indicates 1-sparse code.

    Args:
        m: Nonnegative magnitude tensor, shape (..., n)
        dim: Dimension along which to compute sparsity (default: -1)
        eps: Small constant for numerical stability

    Returns:
        Hoyer sparsity scores, shape (...,)
    """
    n = m.shape[dim]
    if n == 1:
        return torch.zeros_like(m.sum(dim=dim))

    l1_norm = m.sum(dim=dim)
    l2_norm = torch.sqrt((m**2).sum(dim=dim) + eps)

    # Handle zero vectors: return 0 sparsity
    zero_mask = l2_norm < eps
    ratio = torch.where(zero_mask, torch.zeros_like(l1_norm), l1_norm / l2_norm)

    sqrt_n = torch.sqrt(torch.tensor(n, dtype=m.dtype, device=m.device))
    hoyer = (sqrt_n - ratio) / (sqrt_n - 1.0)

    # Clamp to [0, 1] for numerical stability
    return torch.clamp(hoyer, 0.0, 1.0)


def sparsity_rate(m: Tensor, eps: float = 1e-6, dim: int = -1) -> Tensor:
    """
    Compute active fraction (sparsity rate) as approximation of number of active units.

    ρ_ε = (1/n) · Σ 1[|m_i| > ε]

    Args:
        m: Magnitude tensor, shape (..., n)
        eps: Threshold for considering a unit active (default: 1e-6)
        dim: Dimension along which to compute rate (default: -1)

    Returns:
        Active fraction in [0,1], shape (...,)
    """
    n = m.shape[dim]
    active_count = (m > eps).sum(dim=dim).float()
    return active_count / n


def corr_offdiag(acts: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute mean absolute off-diagonal of row-normalized activation Gram (correlation) matrix.

    Lower values indicate better decorrelation/separation.

    Args:
        acts: Activation tensor, shape (batch, n_units)
        eps: Small constant for numerical stability

    Returns:
        Scalar mean absolute off-diagonal correlation
    """
    if acts.ndim != 2:
        raise ValueError(f"Expected 2D tensor (batch, units), got shape {acts.shape}")

    batch_size, n_units = acts.shape
    if batch_size < 2:
        return torch.tensor(0.0, device=acts.device)

    # Row-normalize: each sample has unit L2 norm
    acts_norm = acts / (torch.norm(acts, dim=1, keepdim=True) + eps)

    # Compute Gram (correlation) matrix: G[i,j] = <acts_i, acts_j>
    gram = torch.matmul(acts_norm, acts_norm.t())  # (batch, batch)

    # Extract off-diagonal elements
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=acts.device)
    offdiag = gram[mask]

    return offdiag.abs().mean()


def masked_mse(y_hat: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    """
    Compute MSE only over masked indices.

    Args:
        y_hat: Predictions, shape (batch, ...) or (batch, 1, H, W)
        y: Targets, shape (batch, ...) or (batch, 1, H, W)
        mask: Binary mask (1 = compute, 0 = ignore), same shape as y

    Returns:
        Scalar MSE averaged over masked elements
    """
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=y.device)

    mse_per_element = (y_hat - y) ** 2
    masked_sum = (mse_per_element * mask).sum()
    num_elements = mask.sum()

    return masked_sum / num_elements


def masked_bce(y_hat: Tensor, y: Tensor, mask: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute BCE only over masked indices (FCMT protocol).

    Args:
        y_hat: Predictions (logits or probabilities), shape (batch, ...)
        y: Targets, shape (batch, ...)
        mask: Binary mask (1 = visible/compute, 0 = hidden/ignore), same shape
        eps: Small constant for numerical stability

    Returns:
        Scalar BCE averaged over visible elements
    """
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=y.device)

    # Ensure y_hat is in [0,1] range (apply sigmoid if needed)
    if y_hat.min() < 0 or y_hat.max() > 1:
        y_hat = torch.sigmoid(y_hat)

    # BCE per element
    bce_per_element = -(y * torch.log(y_hat + eps) + (1 - y) * torch.log(1 - y_hat + eps))

    masked_sum = (bce_per_element * mask).sum()
    num_elements = mask.sum()

    return masked_sum / num_elements


def linear_cka(X: Tensor, Y: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Compute Linear Centered Kernel Alignment (CKA) between two activation matrices.

    CKA is invariant to orthogonal transforms and isotropic rescaling, measuring
    whether two representations span similar subspaces.

    Args:
        X: Activations from first layer, shape (batch, n_units_X)
        Y: Activations from second layer, shape (batch, n_units_Y)
        eps: Small constant for numerical stability

    Returns:
        Scalar CKA score in [0,1], where 1 indicates perfect alignment
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Batch sizes must match: X {X.shape[0]} vs Y {Y.shape[0]}")

    batch_size = X.shape[0]
    if batch_size < 2:
        return torch.tensor(1.0, device=X.device)

    # Center features: subtract mean along batch dimension
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)

    # Compute centered Gram matrices
    K_X = torch.matmul(X_centered, X_centered.t())  # (batch, batch)
    K_Y = torch.matmul(Y_centered, Y_centered.t())  # (batch, batch)

    # HSIC (Hilbert-Schmidt Independence Criterion) components
    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum()
    hsic_yy = (K_Y * K_Y).sum()

    # CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    denominator = torch.sqrt(hsic_xx * hsic_yy + eps)
    cka = hsic_xy / denominator

    return torch.clamp(cka, 0.0, 1.0)
