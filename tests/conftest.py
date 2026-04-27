"""Pytest configuration and shared fixtures for all tests."""

import numpy as np
import pytest
import torch


# ────────────────────────────────────────────────────────────────────────────
# Dimension constants shared across tests
# ────────────────────────────────────────────────────────────────────────────
BATCH = 2
SEQ_LEN = 5
CHANNELS = 8
HEIGHT = 10
WIDTH = 12
FUTURE_SEQ = 2


@pytest.fixture
def device():
    """Return CPU device (tests run on CPU regardless of GPU availability)."""
    return torch.device("cpu")


@pytest.fixture
def random_input_sequence():
    """Return a random (B, T, C, H, W) tensor suitable for ConvLSTM input."""
    return torch.randn(BATCH, SEQ_LEN, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def random_target_sequence():
    """Return a random (B, T, C, H, W) target tensor for loss computation."""
    return torch.randn(BATCH, FUTURE_SEQ, CHANNELS, HEIGHT, WIDTH)


@pytest.fixture
def random_numpy_sequence():
    """Return a random (1, T, C, H, W) numpy array for model manager tests."""
    return np.random.randn(1, SEQ_LEN, CHANNELS, HEIGHT, WIDTH).astype(np.float32)


@pytest.fixture
def random_data_array():
    """Return (Time, C, H, W) numpy array for data-loader tests."""
    return np.random.randn(30, CHANNELS, HEIGHT, WIDTH).astype(np.float32)


@pytest.fixture
def convlstm_model(device):
    """Return a small HeatwaveConvLSTM suitable for fast unit tests."""
    from src.models.convlstm import HeatwaveConvLSTM

    model = HeatwaveConvLSTM(
        input_dim=CHANNELS,
        hidden_dim=[8, 8],
        kernel_size=[(3, 3), (3, 3)],
        num_layers=2,
    ).to(device)
    model.eval()
    return model
