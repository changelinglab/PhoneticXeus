"""Linear Projection."""

from typing import Tuple
import torch


class LinearProjection(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.output_dim = output_size
        self.linear_out = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.linear_out(self.dropout(input))
        return output, input_lengths  # no state in this layer

    def output_size(self) -> int:
        return self.output_dim
