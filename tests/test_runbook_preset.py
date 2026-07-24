"""Runbook preset tests."""

from __future__ import annotations

import pytest

from yolo_agent.core.runbook_preset import load_runbook_preset
from yolo_agent.resources import ResourcePaths


def test_coco_yolo26_preset_loads_and_resolves_paths() -> None:
    """The bundled COCO YOLO26 runbook preset should hide harness wiring paths."""
    preset = load_runbook_preset()

    assert preset.name == "coco_yolo26_auto"
    assert preset.default_profile == "debug"
    assert preset.allowed_profiles == ["debug", "pilot", "baseline_full", "baseline_confirm", "candidate_full"]
    assert preset.training_config == ResourcePaths.YOLO26_COCO_GOAL
    assert preset.loop_policy == ResourcePaths.LOOP_POLICY
    assert preset.components == ResourcePaths.COMPONENTS_DIR
    assert preset.search_space == ResourcePaths.SEARCH_SPACE
    assert preset.dataset_manifest_mode == "metadata"


def test_preset_rejects_unknown_profile() -> None:
    """Preset profiles should be an explicit menu, not arbitrary strings."""
    preset = load_runbook_preset()

    with pytest.raises(ValueError, match="not allowed"):
        preset.require_profile("full_send")  # type: ignore[arg-type]
