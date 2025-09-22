from typing import Any, Optional

from torch import Tensor, nn


class Layer(nn.Module):
    def __init__(self, synapses: nn.Module, activation: Optional[nn.Module] = None):
        super().__init__()
        self.register_buffer("currents", None)  # Starts without current values
        self.register_buffer("neurons", None)  # Starts without activation values
        self.synapses = synapses
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        self.currents = self.synapses(input)
        self.neurons = self.activation(self.currents)
        return self.neurons


class GrowingLayer(Layer):
    def __init__(self, init: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.active_units: int = int(init)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        # TODO: FIx as post-synaptic layer might have fixed in_features
        return output[: self.active_units]

    def grow(self, n: int) -> None:
        self.active_units = self.active_units + int(n)


if __name__ == "__main__":
    pass
