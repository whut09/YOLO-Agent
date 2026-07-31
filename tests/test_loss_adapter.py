"""Ultralytics bbox loss adapter tests."""

from __future__ import annotations

import pytest

from yolo_agent.adapters.ultralytics.loss_adapter import (
    CIoULossAdapter,
    UnavailableLossError,
    default_loss_registry,
)


def test_loss_registry_can_lookup_default_adapters() -> None:
    """The default registry should expose only executable bbox losses."""
    registry = default_loss_registry()

    assert registry.names() == ["ciou"]
    assert isinstance(registry.get("ciou"), CIoULossAdapter)


@pytest.mark.parametrize("loss_name", ["wiou", "mpdiou", "nwd"])
def test_unimplemented_losses_are_not_runtime_selectable(loss_name: str) -> None:
    """Unverified losses must not appear in the executable registry."""
    registry = default_loss_registry()

    with pytest.raises(UnavailableLossError, match="verified runtime adapter"):
        registry.get(loss_name)


def test_loss_availability_separates_runtime_and_metadata_entries() -> None:
    registry = default_loss_registry()

    statuses = {item.name: item for item in registry.availability()}
    assert statuses["ciou"].implementation_status == "runtime_integrated"
    assert statuses["ciou"].executable is True
    for name in ("wiou", "mpdiou", "nwd"):
        assert statuses[name].implementation_status == "adapter_required"
        assert statuses[name].executable is False


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
