"""Compatibility checker tests."""

from __future__ import annotations

from pathlib import Path

from yolo_agent.components.compatibility import CompatibilityChecker
from yolo_agent.components.registry import load_cards
from yolo_agent.components.schema import ComponentCard
from yolo_agent.core.task_spec import TaskSpec


ROOT = Path(__file__).resolve().parents[1]
COMPONENT_DIR = ROOT / "configs" / "components"
SCENARIO_DIR = ROOT / "configs" / "scenarios"


def _cards_by_id() -> dict[str, ComponentCard]:
    return {card.id: card for card in load_cards(COMPONENT_DIR)}


def test_checker_accepts_compatible_combination() -> None:
    """A simple supported component set should pass without warnings."""
    cards = _cards_by_id()
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "industrial_defect.yaml")
    checker = CompatibilityChecker()

    result = checker.check(
        task_spec=task_spec,
        base_model={
            "name": "yolov8s",
            "framework": "ultralytics",
            "model_family": "yolov8",
            "export_format": "none",
            "estimated_latency_ms": 30,
            "estimated_model_size_mb": 15,
        },
        components=[cards["loss.bbox.ciou"]],
    )

    assert result.ok is True
    assert result.errors == []
    assert result.warnings == []
    assert result.estimated_risk == "low"


def test_checker_rejects_incompatible_combination() -> None:
    """Framework and model-family mismatches should produce errors."""
    cards = _cards_by_id()
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "edge_realtime.yaml")
    yolov11_only_head = ComponentCard.model_validate(
        {
            "id": "head.yolov11_only",
            "name": "YOLOv11 Only Head",
            "type": "head",
            "compatible_frameworks": ["ultralytics"],
            "compatible_tasks": ["detect"],
            "compatible_model_families": ["yolov11"],
        }
    )
    checker = CompatibilityChecker()

    result = checker.check(
        task_spec=task_spec,
        base_model={
            "name": "yolov5n",
            "framework": "ultralytics",
            "model_family": "yolov5",
        },
        components=[cards["assigner.stal"], yolov11_only_head],
    )

    assert result.ok is False
    assert result.estimated_risk == "high"
    assert any("framework=ultralytics" in error for error in result.errors)
    assert any("model_family=yolov5" in error for error in result.errors)


def test_checker_allows_risky_combination_with_warnings() -> None:
    """Risky but runnable components should return warnings, not errors."""
    cards = _cards_by_id()
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "infrared_small_target.yaml")
    dfl_free_head = ComponentCard.model_validate(
        {
            "id": "head.dfl_free",
            "name": "DFL-Free Head",
            "type": "head",
            "description": "Test-only metadata card for DFL-free head compatibility rules.",
            "target_problems": ["edge_realtime"],
            "compatible_frameworks": ["ultralytics"],
            "compatible_tasks": ["detect"],
            "compatible_model_families": ["yolov8"],
            "constraints": {"export_safe": True},
        }
    )
    checker = CompatibilityChecker()

    result = checker.check(
        task_spec=task_spec,
        base_model={
            "name": "yolov8s",
            "framework": "ultralytics",
            "model_family": "yolov8",
            "export_format": "onnx",
            "estimated_latency_ms": 20,
            "estimated_model_size_mb": 10,
        },
        components=[
            cards["loss.bbox.nwd"],
            cards["head.p2_small_object"],
            dfl_free_head,
        ],
    )

    assert result.ok is True
    assert result.errors == []
    assert result.estimated_risk == "medium"
    assert any("DFL-free head" in warning for warning in result.warnings)
    assert any("P2 small-object heads increase" in warning for warning in result.warnings)
    assert any("max_latency_ms=40.0" in warning for warning in result.warnings)


def test_checker_flags_export_blockers() -> None:
    """Export-specific rules should flag architecture patches for TensorRT."""
    cards = _cards_by_id()
    task_spec = TaskSpec.from_yaml(SCENARIO_DIR / "infrared_small_target.yaml")
    checker = CompatibilityChecker()

    result = checker.check(
        task_spec=task_spec,
        base_model={
            "name": "yolov8s",
            "framework": "ultralytics",
            "model_family": "yolov8",
            "export_format": "tensorrt",
        },
        components=[cards["neck.fullpad"]],
    )

    assert result.ok is False
    assert result.estimated_risk == "high"
    assert any("channel adapter" in warning for warning in result.warnings)
    assert any("TensorRT export may fail" in error for error in result.errors)
