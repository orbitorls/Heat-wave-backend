"""Unit tests for ConvLSTMCell, HeatwaveConvLSTM, and PhysicsInformedLoss."""

import pytest
import torch
import torch.nn as nn

from src.models.convlstm import ConvLSTMCell, HeatwaveConvLSTM, PhysicsInformedLoss


# ────────────────────────────────────────────────────────────────────────────
# ConvLSTMCell
# ────────────────────────────────────────────────────────────────────────────


class TestConvLSTMCell:
    def test_output_shape(self):
        B, C_in, C_hidden, H, W = 2, 4, 8, 6, 6
        cell = ConvLSTMCell(input_dim=C_in, hidden_dim=C_hidden, kernel_size=(3, 3))
        x = torch.randn(B, C_in, H, W)
        h, c = cell(x, cur_state=None)
        assert h.shape == (B, C_hidden, H, W), f"h shape mismatch: {h.shape}"
        assert c.shape == (B, C_hidden, H, W), f"c shape mismatch: {c.shape}"

    def test_with_explicit_state(self):
        B, C_in, C_hidden, H, W = 2, 4, 8, 6, 6
        cell = ConvLSTMCell(input_dim=C_in, hidden_dim=C_hidden, kernel_size=(3, 3))
        x = torch.randn(B, C_in, H, W)
        h0 = torch.zeros(B, C_hidden, H, W)
        c0 = torch.zeros(B, C_hidden, H, W)
        h, c = cell(x, cur_state=(h0, c0))
        assert h.shape == (B, C_hidden, H, W)

    def test_init_hidden_zeros(self):
        cell = ConvLSTMCell(input_dim=3, hidden_dim=6, kernel_size=(3, 3))
        h, c = cell.init_hidden(batch_size=2, image_size=(8, 8))
        assert torch.all(h == 0), "Initial h should be all zeros"
        assert torch.all(c == 0), "Initial c should be all zeros"
        assert h.shape == (2, 6, 8, 8)

    def test_no_bias(self):
        cell = ConvLSTMCell(input_dim=3, hidden_dim=6, kernel_size=(3, 3), bias=False)
        assert cell.conv.bias is None

    def test_gradients_flow(self):
        cell = ConvLSTMCell(input_dim=4, hidden_dim=8, kernel_size=(3, 3))
        x = torch.randn(1, 4, 6, 6, requires_grad=True)
        h, c = cell(x, None)
        loss = h.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow to input"

    def test_different_kernel_sizes(self):
        for k in [1, 3, 5]:
            cell = ConvLSTMCell(input_dim=4, hidden_dim=8, kernel_size=(k, k))
            x = torch.randn(1, 4, 10, 10)
            h, c = cell(x, None)
            assert h.shape == (1, 8, 10, 10), f"Shape mismatch for kernel {k}"

    def test_stateful_update(self):
        """h/c should differ between time steps."""
        cell = ConvLSTMCell(input_dim=4, hidden_dim=8, kernel_size=(3, 3))
        x = torch.randn(1, 4, 6, 6)
        h1, c1 = cell(x, None)
        h2, c2 = cell(x, (h1, c1))
        assert not torch.allclose(h1, h2), "State should update across time steps"


# ────────────────────────────────────────────────────────────────────────────
# HeatwaveConvLSTM
# ────────────────────────────────────────────────────────────────────────────


class TestHeatwaveConvLSTM:
    def _make_model(self, input_dim=4, hidden=[8, 8], num_layers=2):
        return HeatwaveConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden,
            kernel_size=[(3, 3)] * num_layers,
            num_layers=num_layers,
        ).eval()

    def test_output_shape_single_step(self, random_input_sequence):
        model = self._make_model(input_dim=8, hidden=[8, 8], num_layers=2)
        x = random_input_sequence  # (B, T, C, H, W) from conftest
        out = model(x, future_seq=1)
        B, T, C, H, W = x.shape
        assert out.shape == (B, 1, C, H, W), f"Output shape mismatch: {out.shape}"

    def test_output_shape_multi_step(self, random_input_sequence):
        model = self._make_model(input_dim=8, hidden=[8, 8], num_layers=2)
        x = random_input_sequence
        future = 3
        out = model(x, future_seq=future)
        B, T, C, H, W = x.shape
        assert out.shape == (B, future, C, H, W)

    def test_single_layer(self):
        model = HeatwaveConvLSTM(
            input_dim=4,
            hidden_dim=[8],
            kernel_size=[(3, 3)],
            num_layers=1,
        ).eval()
        x = torch.randn(1, 3, 4, 6, 6)
        out = model(x, future_seq=2)
        assert out.shape == (1, 2, 4, 6, 6)

    def test_preserves_spatial_dims(self):
        for H, W in [(8, 8), (16, 12), (5, 7)]:
            model = self._make_model(input_dim=4, hidden=[8, 8], num_layers=2)
            x = torch.randn(1, 4, 4, H, W)
            out = model(x, future_seq=1)
            assert out.shape[-2:] == (H, W), f"Spatial dims changed: {out.shape}"

    def test_mismatched_layer_config_raises(self):
        with pytest.raises(AssertionError):
            HeatwaveConvLSTM(
                input_dim=4,
                hidden_dim=[8],        # length 1
                kernel_size=[(3, 3), (3, 3)],  # length 2
                num_layers=2,
            )

    def test_no_nan_in_output(self, random_input_sequence):
        model = self._make_model(input_dim=8)
        out = model(random_input_sequence, future_seq=2)
        assert not torch.isnan(out).any(), "Model output should not contain NaN"

    def test_deterministic_in_eval(self, random_input_sequence):
        model = self._make_model(input_dim=8)
        with torch.no_grad():
            out1 = model(random_input_sequence, future_seq=1)
            out2 = model(random_input_sequence, future_seq=1)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_gradients_backprop(self, random_input_sequence):
        model = self._make_model(input_dim=8)
        model.train()
        x = random_input_sequence.requires_grad_(True)
        out = model(x, future_seq=1)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None

    def test_parameter_count_nonzero(self):
        model = self._make_model(input_dim=8)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, "Model must have parameters"

    def test_channel_dim_preserved(self):
        """Output channel count must equal input channel count."""
        for C in [4, 8, 16]:
            model = HeatwaveConvLSTM(
                input_dim=C,
                hidden_dim=[16, 16],
                kernel_size=[(3, 3), (3, 3)],
                num_layers=2,
            ).eval()
            x = torch.randn(1, 5, C, 8, 8)
            out = model(x, future_seq=1)
            assert out.shape[2] == C, f"Channel mismatch for C={C}: {out.shape}"


# ────────────────────────────────────────────────────────────────────────────
# PhysicsInformedLoss
# ────────────────────────────────────────────────────────────────────────────


class TestPhysicsInformedLoss:
    def _make_tensors(self, B=2, T=3, C=4, H=8, W=8):
        pred = torch.randn(B, T, C, H, W)
        target = torch.randn(B, T, C, H, W)
        return pred, target

    def test_returns_three_values(self):
        loss_fn = PhysicsInformedLoss()
        pred, target = self._make_tensors()
        total, mse, phy = loss_fn(pred, target)
        assert total is not None
        assert mse is not None
        assert phy is not None

    def test_total_equals_mse_plus_lambda_phy(self):
        lam = 0.2
        loss_fn = PhysicsInformedLoss(lambda_phy=lam)
        pred, target = self._make_tensors()
        total, mse, phy = loss_fn(pred, target)
        expected = mse + lam * phy
        assert torch.allclose(total, expected, atol=1e-6), (
            f"total={total.item():.6f}, mse+lam*phy={expected.item():.6f}"
        )

    def test_perfect_prediction_zero_mse(self):
        loss_fn = PhysicsInformedLoss(lambda_phy=0.0)
        pred = torch.randn(2, 3, 4, 8, 8)
        total, mse, _ = loss_fn(pred, pred)
        assert mse.item() < 1e-6, f"MSE should be ~0 for identical tensors: {mse.item()}"

    def test_zero_lambda_ignores_physics(self):
        loss_fn_no_phy = PhysicsInformedLoss(lambda_phy=0.0)
        loss_fn_with_phy = PhysicsInformedLoss(lambda_phy=1.0)
        pred, target = self._make_tensors()
        total_no, mse_no, _ = loss_fn_no_phy(pred, target)
        total_with, mse_with, _ = loss_fn_with_phy(pred, target)
        # MSE component should be identical regardless of lambda
        assert torch.allclose(mse_no, mse_with, atol=1e-6)

    def test_loss_is_scalar(self):
        loss_fn = PhysicsInformedLoss()
        pred, target = self._make_tensors()
        total, mse, phy = loss_fn(pred, target)
        assert total.ndim == 0, "Total loss must be a scalar"
        assert mse.ndim == 0
        assert phy.ndim == 0

    def test_loss_is_non_negative(self):
        loss_fn = PhysicsInformedLoss()
        pred, target = self._make_tensors()
        total, mse, phy = loss_fn(pred, target)
        assert total.item() >= 0
        assert mse.item() >= 0
        assert phy.item() >= 0

    def test_gradients_flow_through_loss(self):
        loss_fn = PhysicsInformedLoss()
        pred = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        target = torch.randn(2, 3, 4, 8, 8)
        total, _, _ = loss_fn(pred, target)
        total.backward()
        assert pred.grad is not None

    def test_default_lambda(self):
        loss_fn = PhysicsInformedLoss()
        assert loss_fn.lambda_phy == 0.1

    def test_4d_input(self):
        """Loss should handle 4D (B, C, H, W) tensors as well."""
        loss_fn = PhysicsInformedLoss()
        pred = torch.randn(2, 4, 8, 8)
        target = torch.randn(2, 4, 8, 8)
        total, mse, phy = loss_fn(pred, target)
        assert total.ndim == 0
