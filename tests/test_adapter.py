"""Ultralytics adapter boundary tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.adapters.ultralytics.adapter import UltralyticsAdapter
from yolo_agent.agents.candidate_generator import CandidateConfig
from yolo_agent.resources import ResourcePaths


def _candidate() -> CandidateConfig:
    return CandidateConfig(
        candidate_id="baseline",
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
    )


def test_adapter_is_available_returns_false_when_ultralytics_missing() -> None:
    """Adapter should report unavailability when the package is not installed."""
    adapter = UltralyticsAdapter()
    assert adapter.is_available() is False


def test_adapter_available_losses_returns_default_names() -> None:
    """Adapter should expose the default loss registry names."""
    adapter = UltralyticsAdapter()
    names = adapter.available_losses()
    assert "ciou" in names
    assert "wiou" in names
    assert "mpdiou" in names
    assert "nwd" in names


def test_adapter_get_loss_returns_buildable_ciou() -> None:
    """Adapter.get_loss should return a configurable CIoU adapter."""
    adapter = UltralyticsAdapter()
    loss = adapter.get_loss("ciou")
    assert loss.name == "ciou"
    same = loss.build({"weight": 0.5})
    assert same.config == {"weight": 0.5}


def test_adapter_generate_model_yaml_writes_output(tmp_path: Path) -> None:
    """Adapter should generate a model YAML draft for a candidate."""
    adapter = UltralyticsAdapter()
    result = adapter.generate_model_yaml(
        candidate=_candidate(),
        base_template=ResourcePaths.ULTRALYTICS_BASE_TEMPLATE,
        output_dir=tmp_path / "models",
        nc=2,
        dry_run=False,
    )

    assert result.output_path == tmp_path / "models" / "baseline.yaml"
    assert result.output_path.exists()
    assert "set nc=2" in result.changes
    assert result.warnings == []

    with result.output_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    assert data["nc"] == 2


def test_adapter_generate_model_yaml_dry_run_returns_result(tmp_path: Path) -> None:
    """Adapter dry-run should not write files but still return changes."""
    adapter = UltralyticsAdapter()
    result = adapter.generate_model_yaml(
        candidate=_candidate(),
        base_template=ResourcePaths.ULTRALYTICS_BASE_TEMPLATE,
        output_dir=tmp_path / "models",
        dry_run=True,
    )

    assert result.dry_run is True
    assert result.output_path.exists() is False
    assert any("copy baseline template" in change for change in result.changes)


def test_adapter_build_train_command_returns_yolo_string() -> None:
    """Adapter should assemble a basic yolo train command string."""
    adapter = UltralyticsAdapter()
    node = type("Node", (), {
        "node_id": "node-1",
        "candidate_config": _candidate(),
    })()
    cmd = adapter.build_train_command(node)

    assert cmd == "yolo train model=generated_models/baseline.yaml"


def test_adapter_build_train_command_allows_overrides() -> None:
    """Adapter should accept explicit command and model yaml overrides."""
    adapter = UltralyticsAdapter()
    node = type("Node", (), {
        "node_id": "node-1",
        "candidate_config": _candidate(),
    })()
    cmd = adapter.build_train_command(
        node,
        command="yolo train model=/tmp/custom.yaml epochs=50",
        model_yaml_path="/tmp/custom.yaml",
    )

    assert cmd == "yolo train model=/tmp/custom.yaml epochs=50"


def test_adapter_smoke_check_returns_false_without_ultralytics() -> None:
    """Adapter smoke_check should fail gracefully when ultralytics is missing."""
    adapter = UltralyticsAdapter()
    assert adapter.smoke_check("nonexistent.yaml") is False
