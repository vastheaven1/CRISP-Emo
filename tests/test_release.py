from pathlib import Path
import sys

import numpy as np
import pytest
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from evaluation import summarize_window_predictions  # noqa: E402
from losses import reliable_logit_full_geometry_loss  # noqa: E402
from model import CRISPEmo  # noqa: E402


def test_model_shape_and_parameter_count() -> None:
    torch.manual_seed(0)
    model = CRISPEmo(
        eeg_channels=10, embed_dim=64, fusion_layers=1, modality_dropout=0.10
    ).eval()
    inputs = (
        torch.randn(2, 10, 640),
        torch.randn(2, 1, 160),
        torch.randn(2, 1, 40),
        torch.randn(2, 2, 1280),
        torch.tensor([0, 1]),
    )
    with torch.no_grad():
        output = model(*inputs)
    assert output["logits"].shape == (2, 2, 2)
    assert output["fused_embedding"].shape == (2, 64)
    assert model.parameter_count() == 162_660


def test_formal_peer_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(7)
    outputs = (
        {
            "logits": torch.randn(6, 2, 2, requires_grad=True),
            "fused_embedding": torch.randn(6, 8, requires_grad=True),
        },
        {
            "logits": torch.randn(6, 2, 2, requires_grad=True),
            "fused_embedding": torch.randn(6, 8, requires_grad=True),
        },
    )
    labels = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1], [0, 1], [1, 0]])
    domains = torch.tensor([0, 0, 0, 1, 1, 1])
    weights = torch.tensor([1.0, 0.5, 1.0, 1.0, 0.5, 1.0])

    logit_loss, feature_loss, diagnostics = reliable_logit_full_geometry_loss(
        outputs, labels, domains, weights
    )
    total = logit_loss + feature_loss
    assert torch.isfinite(total)
    assert float(logit_loss.detach()) >= -1e-6
    assert float(feature_loss.detach()) >= -1e-6
    total.backward()
    gradients = [item[key].grad for item in outputs for key in ("logits", "fused_embedding")]
    assert any(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert diagnostics["winner0"] + diagnostics["winner1"] == pytest.approx(1.0)
    assert 0.0 <= diagnostics["reliable_coverage"] <= 1.0


def test_fitted_thresholds_are_reused_without_metric_drift() -> None:
    labels = np.asarray(
        [[0, 1], [0, 0], [1, 1], [1, 0], [0, 0], [1, 0], [0, 1], [1, 1]]
    )
    probabilities = np.asarray(
        [[0.1, 0.8], [0.4, 0.3], [0.7, 0.9], [0.6, 0.2],
         [0.2, 0.4], [0.8, 0.3], [0.3, 0.7], [0.9, 0.6]]
    )
    domains = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    fitted = summarize_window_predictions(
        probabilities, labels, domains, fit_thresholds=True,
        threshold_objective="macro_f1", threshold_scope="dataset",
    )
    frozen = summarize_window_predictions(
        probabilities, labels, domains,
        decision_thresholds=fitted["decision_thresholds"],
        threshold_objective="macro_f1", threshold_scope="dataset",
    )
    assert frozen["decision_thresholds"] == fitted["decision_thresholds"]
    assert frozen["selection_scores"] == pytest.approx(fitted["selection_scores"])
