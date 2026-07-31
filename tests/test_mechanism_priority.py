from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from yolo_agent.research.mechanism_priority import MechanismPriorityConfig


def test_default_priorities_put_requested_mechanisms_first() -> None:
    config = MechanismPriorityConfig.from_yaml()

    assert config.priority_for("sampling.small_object").family_id == "small_object"  # type: ignore[union-attr]
    assert config.priority_for("loss.quality.correlation").family_id == "quality_alignment"  # type: ignore[union-attr]
    assert config.priority_for("distillation.yolo26_teacher_student").priority_rank == 3  # type: ignore[union-attr]
    assert config.priority_for("neck.multi_scale_fusion").family_id == "multi_scale_neck"  # type: ignore[union-attr]
    assert config.priority_for("optimizer.adamw") is None
    assert config.unresolved_reason("small-object") == (
        "task_scope_not_canonical_mechanism"
    )
    assert config.unresolved_reason("DETR") == "detector_family_label_not_component"
    assert config.unresolved_reason("specific_new_loss") == (
        "canonical_component_mapping_required"
    )
    assert config.is_separate_detector_family("oriented-detr") is True


def test_priority_config_rejects_duplicate_component_ownership(tmp_path: Path) -> None:
    path = tmp_path / "priority.yaml"
    path.write_text(
        """schema_version: research_priority.v1
mechanism_families:
  - family_id: first
    priority_rank: 1
    canonical_component_ids: [same.component]
  - family_id: second
    priority_rank: 2
    canonical_component_ids: [same.component]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="belongs to both"):
        MechanismPriorityConfig.from_yaml(path)
