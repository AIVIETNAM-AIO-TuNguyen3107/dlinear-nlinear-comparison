"""Smoke tests for src.models (EXECUTION_PLAN §2.1 shapes / param counts)."""

import pytest
import torch

from src.models import DLinear, Linear, MovingAvg, NLinear


@pytest.fixture
def shapes():
    # B, C, L lookback; H = pred_len
    return dict(B=4, C=3, L=16, H=8)


def test_moving_avg_preserves_length(shapes):
    B, C, L = shapes["B"], shapes["C"], shapes["L"]
    x = torch.randn(B, C, L)
    y = MovingAvg(kernel=3)(x)
    assert y.shape == (B, C, L)


def test_linear_shared_forward_shape(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = Linear(seq_len=L, pred_len=H, individual=False)
    y = model(torch.randn(B, C, L))
    assert y.shape == (B, C, H)


def test_linear_individual_forward_shape(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = Linear(seq_len=L, pred_len=H, n_vars=C, individual=True)
    y = model(torch.randn(B, C, L))
    assert y.shape == (B, C, H)


def test_linear_shared_param_count_approx_LT(shapes):
    """Shared W: ≈ L·H weights (ignore bias)."""
    L, H, _ = shapes["L"], shapes["H"], shapes["C"]
    model = Linear(seq_len=L, pred_len=H, individual=False)
    n_weight = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    assert n_weight == L * H


def test_linear_individual_param_count_approx_CLT(shapes):
    L, H, C = shapes["L"], shapes["H"], shapes["C"]
    model = Linear(seq_len=L, pred_len=H, n_vars=C, individual=True)
    n_weight = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    assert n_weight == C * L * H


def test_dlinear_shared_forward_shape(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = DLinear(seq_len=L, pred_len=H, kernel=3, individual=False)
    y = model(torch.randn(B, C, L))
    # nn.Linear on last dim → (B, C, H); comment in models.py saying B H C is stale
    assert y.shape == (B, C, H)


def test_dlinear_individual_forward_shape(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = DLinear(seq_len=L, pred_len=H, n_vars=C, kernel=3, individual=True)
    y = model(torch.randn(B, C, L))
    assert y.shape == (B, C, H)


def test_dlinear_shared_param_count_approx_2LT(shapes):
    """Shared W: ≈ 2·L·H weights (ignore bias)."""
    L, H, _ = shapes["L"], shapes["H"], shapes["C"]
    model = DLinear(seq_len=L, pred_len=H, individual=False)
    n_weight = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    assert n_weight == 2 * L * H


def test_dlinear_individual_param_count_approx_2CLT(shapes):
    L, H, C = shapes["L"], shapes["H"], shapes["C"]
    model = DLinear(seq_len=L, pred_len=H, n_vars=C, individual=True)
    n_weight = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    assert n_weight == 2 * C * L * H


def test_nlinear_shared_param_count_approx_LH(shapes):
    """Shared W: L·H weights (ignore bias)."""
    L, H, _ = shapes["L"], shapes["H"], shapes["C"]
    model = NLinear(seq_len=L, pred_len=H, individual=False)
    n_weight = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    assert n_weight == L * H


def test_nlinear_individual_param_count_approx_CLH(shapes):
    L, H, C = shapes["L"], shapes["H"], shapes["C"]
    model = NLinear(seq_len=L, pred_len=H, n_vars=C, individual=True)
    n_weight = sum(p.numel() for n, p in model.named_parameters() if "bias" not in n)
    assert n_weight == C * L * H


def test_nlinear_shared_forward_shape(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    y = NLinear(seq_len=L, pred_len=H)(torch.randn(B, C, L))
    assert y.shape == (B, C, H)


def _zero_linear_(lin: torch.nn.Linear) -> None:
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()


def test_nlinear_zero_weights_predicts_last_lookback(shapes):
    """y_hat' = 0 ⇒ y_hat = last lookback value (add-back)."""
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = NLinear(seq_len=L, pred_len=H, individual=False)
    _zero_linear_(model.Linear)
    x = torch.randn(B, C, L)
    y = model(x)
    expected = x[:, :, -1:].expand(B, C, H)
    assert y.shape == (B, C, H)
    assert torch.allclose(y, expected)


def test_nlinear_individual_zero_weights_predicts_last_lookback(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = NLinear(seq_len=L, pred_len=H, n_vars=C, individual=True)
    for lin in model.Linear:
        _zero_linear_(lin)
    x = torch.randn(B, C, L)
    y = model(x)
    expected = x[:, :, -1:].expand(B, C, H)
    assert torch.allclose(y, expected)


def test_nlinear_level_shift_equivariant(shapes):
    """Shift whole series by c ⇒ forecast shifts by c (recentering)."""
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = NLinear(seq_len=L, pred_len=H, individual=False)
    x = torch.randn(B, C, L)
    c = 7.5
    assert torch.allclose(model(x + c), model(x) + c, atol=1e-5)


def test_nlinear_individual_level_shift_equivariant(shapes):
    B, C, L, H = shapes["B"], shapes["C"], shapes["L"], shapes["H"]
    model = NLinear(seq_len=L, pred_len=H, n_vars=C, individual=True)
    x = torch.randn(B, C, L)
    c = -3.25
    assert torch.allclose(model(x + c), model(x) + c, atol=1e-5)
