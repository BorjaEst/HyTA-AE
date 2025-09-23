import math
from typing import Any, List, Optional, Tuple

import torch
from torch import Size, Tensor, autograd, nn


# -------------------------------------------------------------------------------------------
def apply(activations: Tensor, fb_weight: Tensor, target: Tensor) -> None:
    """
    Apply DRTP by broadcasting the target through a fixed random matrix and
    backpropagating the resulting local gradient from the post-activation node.

    Parameters
    ----------
    activations: Tensor
        Post-activation tensor to serve as backward root (requires grad and NOT detached).
    fb_weight: Tensor
        Fixed feedback matrix of shape (target_features, units).
    target: Tensor
        Supervised targets of shape (batch, target_features). Detached internally.

    Notes
    -----
    - Autograd will incorporate the activation derivative automatically because
      we backprop from the post-activation node.
    - This performs a backward pass rooted at `activations` only; graphs must be
      decoupled across layers (use detach in forward).
    """
    delta = torch.matmul(target.detach(), fb_weight)  # (batch, units)
    autograd.backward(activations, delta)
    return None


# -------------------------------------------------------------------------------------------
class Linear(nn.Module):
    """
    DRTP projector module. Stores a fixed random projection from targets to a layer's units,
    and triggers a local backward rooted at the given post-activation tensor.
    """

    def __init__(self, target_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        fb_weights = torch.zeros(target_features, out_features, device=device, dtype=dtype)
        self.register_buffer("fb_weight", fb_weights)  # (target_features, units)
        self.reset_feedback()
        self.last_input: Optional[Tensor] = None

    @property
    def target_features(self) -> int:
        return self.fb_weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.fb_weight.shape[1]

    def forward(self, input: Tensor) -> Tensor:
        # Store non-detached activation for backward; return detached for feedforward isolation
        self.last_input = input
        return input.detach()

    def feedback(self, target: Tensor) -> None:
        if self.last_input is None:
            raise RuntimeError("feedback() called before forward; no stored activation.")
        apply(self.last_input, self.fb_weight, target)

    def extra_repr(self) -> str:
        return f"target_features={self.target_features}, out_features={self.out_features}"

    def reset_feedback(self) -> None:
        limit = 1.0 / math.sqrt(self.target_features)
        torch.nn.init.uniform_(self.fb_weight, -limit, limit)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example usage of DRTP in a simple neural network
    print("=== DRTP Layer Example ===")
    torch.manual_seed(0)

    # Network parameters
    input_dim, hidden1_dim, hidden2_dim, output_dim = 5, 4, 3, 2
    batch_size = 2

    # Create network layers
    layer1 = nn.Linear(input_dim, hidden1_dim)
    fb1 = Linear(output_dim, hidden1_dim)  # (target_features=output_dim, units=hidden1_dim)
    layer2 = nn.Linear(hidden1_dim, hidden2_dim)
    fb2 = Linear(output_dim, hidden2_dim)  # (target_features=output_dim, units=hidden2_dim)
    layer3 = nn.Linear(hidden2_dim, output_dim)

    # Optimizer
    parameters = list(layer1.parameters()) + list(layer2.parameters()) + list(layer3.parameters())
    optimizer = torch.optim.SGD(parameters, lr=1e-2)
    optimizer.zero_grad()

    # Create input and target
    x = torch.randn(batch_size, input_dim)
    target = torch.randint(0, 2, (batch_size, output_dim), dtype=torch.float32)  # simple binary targets

    # Forward pass (projector variant)
    h1 = torch.relu(layer1(x))
    h1_ = fb1(h1)  # returns detached view
    h2 = torch.relu(layer2(h1_))
    h2_ = fb2(h2)  # returns detached view
    logits = layer3(h2_)
    output = torch.sigmoid(logits)

    h1.retain_grad()
    h2.retain_grad()

    print("\nForward pass:")
    print(f"  Hidden 1 shape: {h1.shape}")
    print(f"  Hidden 2 shape: {h2.shape}")
    print(f"  Output shape: {output.shape}")

    # Compute loss
    loss = nn.BCELoss()(output, target)
    print(f"  Loss: {loss.item():.6f}")

    # DRTP feedback for hidden layers
    fb1.feedback(target)
    fb2.feedback(target)

    # Standard backward for output layer
    loss.backward()

    print("\nAfter backward:")
    print(f"  Layer1 grad norm: {layer1.weight.grad.norm():.4f}")
    print(f"  Layer2 grad norm: {layer2.weight.grad.norm():.4f}")
    print(f"  Layer3 grad norm: {layer3.weight.grad.norm():.4f}")

    # Verify DRTP projections have expected shapes
    proj1 = torch.matmul(target, fb1.fb_weight)
    proj2 = torch.matmul(target, fb2.fb_weight)
    print("\nDRTP projections:")
    print(f"  DRTP1 projection shape: {proj1.shape} (should match h1 shape)")
    print(f"  DRTP2 projection shape: {proj2.shape} (should match h2 shape)")

    optimizer.step()

    print("\nDRTP Layer example completed successfully!")
