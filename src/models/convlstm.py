"""ConvLSTM model components for heatwave forecasting.

This module defines a ConvLSTM cell, a multi-layer ConvLSTM forecasting
model, and a physics-informed loss used during training.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell with convolutional gates."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cur_state is None:
            batch_size, _, height, width = input_tensor.shape
            cur_state = self.init_hidden(batch_size=batch_size, image_size=(height, width))

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states to zeros."""
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        )


class HeatwaveConvLSTM(nn.Module):
    """Multi-layer ConvLSTM model for sequence-to-sequence forecasting."""

    def __init__(self, input_dim: int, hidden_dim: List[int], kernel_size: List[Tuple[int, int]], num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        assert len(hidden_dim) == num_layers, f"hidden_dim length {len(hidden_dim)} != num_layers {num_layers}"
        assert len(kernel_size) == num_layers, f"kernel_size length {len(kernel_size)} != num_layers {num_layers}"

        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim[i - 1]
            self.encoder_cells.append(ConvLSTMCell(in_dim, hidden_dim[i], kernel_size[i]))

        self.output_conv = nn.Conv2d(hidden_dim[-1], input_dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, future_seq: int = 1) -> torch.Tensor:
        batch_size, seq_len, _, height, width = x.shape

        states = []
        for i in range(self.num_layers):
            cell = self.encoder_cells[i]
            assert isinstance(cell, ConvLSTMCell)
            states.append(cell.init_hidden(batch_size, (height, width)))

        for t in range(seq_len):
            layer_input = x[:, t]
            new_states = []
            for i in range(self.num_layers):
                cell = self.encoder_cells[i]
                assert isinstance(cell, ConvLSTMCell)
                h, c = cell(layer_input, states[i])
                new_states.append((h, c))
                layer_input = h
            states = new_states

        predictions: List[torch.Tensor] = []
        dec_input = self.output_conv(states[-1][0])

        for _ in range(future_seq):
            layer_input = dec_input
            new_states = []
            for i in range(self.num_layers):
                cell = self.encoder_cells[i]
                assert isinstance(cell, ConvLSTMCell)
                h, c = cell(layer_input, states[i])
                new_states.append((h, c))
                layer_input = h
            states = new_states
            dec_input = self.output_conv(states[-1][0])
            predictions.append(dec_input.unsqueeze(1))

        return torch.cat(predictions, dim=1)


class PhysicsInformedLoss(nn.Module):
    """MSE loss with additional spatial-gradient regularization."""

    def __init__(self, lambda_phy: float = 0.1):
        super().__init__()
        self.lambda_phy = lambda_phy
        self.mse = nn.MSELoss()

    def _spatial_gradient_loss(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 5:
            ch_idx = min(1, pred.shape[2] - 1)
            temp = pred[:, :, ch_idx, :, :]
        else:
            ch_idx = min(1, pred.shape[1] - 1)
            temp = pred[:, ch_idx, :, :].unsqueeze(1)

        grad_x = temp[..., :, 1:] - temp[..., :, :-1]
        grad_y = temp[..., 1:, :] - temp[..., :-1, :]
        return torch.mean(grad_x ** 2) + torch.mean(grad_y ** 2)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mse_loss = self.mse(prediction, target)
        physics_loss = self._spatial_gradient_loss(prediction)
        total_loss = mse_loss + self.lambda_phy * physics_loss
        return total_loss, mse_loss, physics_loss


__all__ = ["ConvLSTMCell", "HeatwaveConvLSTM", "PhysicsInformedLoss"]
