"""The objective's baseline comparison protocol is frozen per run.

``build_baseline_protocol_hash`` feeds ``OptimizationObjective.baseline_protocol_hash``,
which every candidate node binds as ``baseline_protocol_hash`` metadata and every
matched control is validated against.  The design contract (see the matched
control plan validator in ``core/round_execution_plan.py``) is that the code
version changes on every fork while the inherited objective comparison hash
stays fixed.  A protocol hash that drifts with agent commits wedges resumed
runs with ``matched_control_protocol_hash_mismatch``.
"""

from __future__ import annotations

from pathlib import Path

from yolo_agent.core.optimization_objective import build_baseline_protocol_hash
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig


def _training_config() -> UltralyticsTrainingConfig:
    return UltralyticsTrainingConfig.from_yaml(
        "configs/training/yolo26_coco_goal.yaml",
        budget_profile="pilot",
    )


def _hash_kwargs() -> dict[str, object]:
    return {
        "model": "yolo26n.pt",
        "data_yaml": Path("configs/datasets/coco.yaml"),
        "training_config": _training_config(),
        "dataset_version": "coco2017",
        "dataset_manifest_sha256": "manifest-sha",
    }


def test_baseline_protocol_hash_ignores_agent_code_version(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Agent commits must not move the baseline comparison protocol."""
    from yolo_agent.core import run_protocol

    baseline = build_baseline_protocol_hash(**_hash_kwargs())
    monkeypatch.setattr(
        run_protocol,
        "current_code_version",
        lambda root=".": "deadbeef" + "0" * 56,
    )
    assert build_baseline_protocol_hash(**_hash_kwargs()) == baseline


def test_baseline_protocol_hash_tracks_ultralytics_version(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The training environment stays part of the comparison protocol."""
    from yolo_agent.core import run_protocol

    baseline = build_baseline_protocol_hash(**_hash_kwargs())
    monkeypatch.setattr(
        run_protocol,
        "installed_ultralytics_version",
        lambda: "0.0.0-test-drift",
    )
    assert build_baseline_protocol_hash(**_hash_kwargs()) != baseline
