"""Ultralytics bbox loss adapter tests."""

from __future__ import annotations

import pytest

from yolo_agent.adapters.ultralytics.loss_adapter import (
    UNVERIFIED_LOSS_MESSAGE,
    CIoULossAdapter,
    default_loss_registry,
)


def test_loss_registry_can_lookup_default_adapters() -> None:
    """The default registry should expose all scaffolded bbox losses."""
    registry = default_loss_registry()

    assert registry.names() == ["ciou", "mpdiou", "nwd", "wiou"]
    assert isinstance(registry.get("ciou"), CIoULossAdapter)


@pytest.mark.parametrize("loss_name", ["wiou", "mpdiou", "nwd"])
def test_unimplemented_losses_raise_clear_error(loss_name: str) -> None:
    """Unverified losses must not be silently usable."""
    torch = pytest.importorskip("torch")
    registry = default_loss_registry()
    adapter = registry.get(loss_name)
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

    with pytest.raises(NotImplementedError, match=UNVERIFIED_LOSS_MESSAGE):
        adapter.compute(pred, target)


def test_ciou_simplified_loss_can_backward() -> None:
    """The simplified CIoU adapter should participate in torch autograd."""
    torch = pytest.importorskip("torch")
    adapter = CIoULossAdapter()
    pred = torch.tensor(
        [
            [0.0, 0.0, 2.0, 2.0],
            [1.0, 1.0, 3.0, 3.0],
        ],
        requires_grad=True,
    )
    target = torch.tensor(
        [
            [0.5, 0.5, 2.5, 2.5],
            [1.0, 1.0, 3.2, 3.2],
        ]
    )

    loss = adapter.compute(pred, target)
    loss.backward()

    assert loss.ndim == 0
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

