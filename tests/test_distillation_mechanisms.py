from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yolo_agent.components.distillation.mechanism_losses import (
    DistillationInputs,
    build_distillation_mechanism_loss,
)
from yolo_agent.components.distillation.mechanisms import (
    DISTILLATION_COMPONENTS,
    DISTILLATION_MECHANISMS,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    YOLO26DistillationAdapter,
    YOLO26DistillationConfig,
    YOLO26DistillationRuntimePlugin,
)
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.contracts import ComponentContract


def test_distillation_mechanisms_have_independent_runtime_identities() -> None:
    assert set(DISTILLATION_MECHANISMS) == {
        "logits",
        "feature",
        "localization",
        "relation",
        "attention",
        "masked_feature",
        "quality_aware",
        "teacher_ensemble",
    }
    assert len(DISTILLATION_COMPONENTS) == 8
    assert len(
        {item.changed_variable for item in DISTILLATION_MECHANISMS.values()}
    ) == 8
    assert all(
        item.changed_variable == f"loss.distillation.{item.mechanism}.weight"
        for item in DISTILLATION_MECHANISMS.values()
    )


def test_distillation_mechanism_requirements_are_explicit() -> None:
    assert DISTILLATION_MECHANISMS["feature"].requires_features
    assert DISTILLATION_MECHANISMS["relation"].requires_features
    assert DISTILLATION_MECHANISMS["attention"].requires_features
    assert DISTILLATION_MECHANISMS["masked_feature"].requires_features
    assert DISTILLATION_MECHANISMS["localization"].requires_boxes
    assert DISTILLATION_MECHANISMS["teacher_ensemble"].requires_multiple_teachers
    assert not DISTILLATION_MECHANISMS["logits"].requires_features


@pytest.mark.parametrize(
    "mechanism", ["logits", "feature", "localization", "relation"]
)
def test_base_distillation_losses_shape_backward_and_amp(mechanism: str) -> None:
    student_logits = torch.randn(2, 4, 7, requires_grad=True)
    student_boxes = torch.randn(2, 4, 7, requires_grad=True)
    student_features = [torch.randn(2, 5, 8, 8, requires_grad=True)]
    teacher_features = [torch.randn(2, 7, 8, 8, requires_grad=True)]
    inputs = DistillationInputs(
        student_logits=student_logits,
        teacher_logits=torch.randn(2, 4, 7, requires_grad=True),
        student_features=student_features,
        teacher_features=teacher_features,
        student_boxes=student_boxes,
        teacher_boxes=torch.randn(2, 4, 7, requires_grad=True),
    )
    options = {"class_dim": 1} if mechanism == "logits" else {}

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = build_distillation_mechanism_loss(mechanism, **options).compute(
            inputs
        )
    output.loss.backward()

    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    if mechanism == "logits":
        assert student_logits.grad is not None
        assert inputs.teacher_logits.grad is None
    elif mechanism == "localization":
        assert student_boxes.grad is not None
        assert inputs.teacher_boxes.grad is None
    else:
        assert student_features[0].grad is not None
        assert teacher_features[0].grad is None


@pytest.mark.parametrize("mechanism", ["attention", "masked_feature"])
def test_attention_distillation_losses_backward_without_teacher_grad(
    mechanism: str,
) -> None:
    student = [torch.randn(2, 5, 8, 8, requires_grad=True)]
    teacher = [torch.randn(2, 9, 6, 6, requires_grad=True)]
    inputs = DistillationInputs(
        student_logits=torch.randn(2, 3, 4),
        teacher_logits=torch.randn(2, 3, 4),
        student_features=student,
        teacher_features=teacher,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = build_distillation_mechanism_loss(mechanism).compute(inputs)
    output.loss.backward()

    assert student[0].grad is not None
    assert teacher[0].grad is None
    assert output.metrics["feature_level_count"] == 1.0


def test_masked_feature_distillation_uses_bounded_teacher_mask() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 2, 4),
        teacher_logits=torch.randn(1, 2, 4),
        student_features=[torch.randn(1, 4, 4, 4)],
        teacher_features=[torch.randn(1, 4, 4, 4)],
    )

    output = build_distillation_mechanism_loss(
        "masked_feature", mask_ratio=0.25
    ).compute(inputs)

    assert output.metrics["masked_position_count"] == 4.0
    assert output.metrics["masked_position_fraction"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="ratio"):
        build_distillation_mechanism_loss("masked_feature", mask_ratio=0.0)


def test_quality_aware_distillation_weights_teacher_confidence() -> None:
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 5, requires_grad=True)
    output = build_distillation_mechanism_loss(
        "quality_aware", class_dim=1
    ).compute(
        DistillationInputs(student_logits=student, teacher_logits=teacher)
    )

    output.loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
    assert 0.0 < output.metrics["mean_teacher_quality"] <= 1.0


def test_teacher_ensemble_averages_multiple_frozen_teacher_profiles() -> None:
    student = torch.randn(2, 3, 5, requires_grad=True)
    teachers = [
        torch.randn(2, 3, 5, requires_grad=True),
        torch.randn(2, 3, 5, requires_grad=True),
    ]
    output = build_distillation_mechanism_loss(
        "teacher_ensemble", class_dim=1
    ).compute(
        DistillationInputs(student_logits=student, teacher_logits=teachers)
    )

    output.loss.backward()

    assert student.grad is not None
    assert all(teacher.grad is None for teacher in teachers)
    assert output.metrics["teacher_count"] == 2.0
    with pytest.raises(ValueError, match="at least two teachers"):
        build_distillation_mechanism_loss("teacher_ensemble").compute(
            DistillationInputs(
                student_logits=torch.randn(2, 3),
                teacher_logits=[torch.randn(2, 3)],
            )
        )


@pytest.mark.parametrize("mechanism", sorted(DISTILLATION_MECHANISMS))
def test_runtime_config_binds_each_mechanism_to_canonical_identity(
    mechanism: str,
) -> None:
    spec = DISTILLATION_MECHANISMS[mechanism]
    values = {
        "mechanism": mechanism,
        "component_id": spec.component_id,
        "changed_variable": spec.changed_variable,
        "teacher": "yolo26s.pt",
        "teachers": ["yolo26m.pt"] if mechanism == "teacher_ensemble" else [],
    }

    config = YOLO26DistillationConfig.model_validate(values)

    assert config.mechanism == mechanism
    assert config.component_id == spec.component_id
    assert config.changed_variable == spec.changed_variable


def test_runtime_config_rejects_unbound_mechanism_and_single_teacher_ensemble() -> None:
    with pytest.raises(ValueError, match="component identity"):
        YOLO26DistillationConfig(
            mechanism="logits",
            component_id="distillation.feature",
            changed_variable="loss.distillation.logits.weight",
        )
    with pytest.raises(ValueError, match="at least two teachers"):
        YOLO26DistillationConfig(
            mechanism="teacher_ensemble",
            component_id="distillation.teacher_ensemble",
            changed_variable="loss.distillation.teacher_ensemble.weight",
        )


@pytest.mark.parametrize("mechanism", sorted(DISTILLATION_MECHANISMS))
def test_runtime_computes_one_weighted_mechanism_without_teacher_grad(
    mechanism: str,
) -> None:
    spec = DISTILLATION_MECHANISMS[mechanism]
    options = {
        "mechanism": mechanism,
        "component_id": spec.component_id,
        "changed_variable": spec.changed_variable,
        "weight": 0.25,
        "teachers": ["yolo26m.pt"] if mechanism == "teacher_ensemble" else [],
    }
    plugin = YOLO26DistillationRuntimePlugin(**options)
    student_scores = torch.randn(2, 3, 5, requires_grad=True)
    student_boxes = torch.randn(2, 4, 5, requires_grad=True)
    student_features = [torch.randn(2, 5, 6, 6, requires_grad=True)]
    teacher_scores = torch.randn(2, 3, 5, requires_grad=True)
    teacher_boxes = torch.randn(2, 4, 5, requires_grad=True)
    teacher_features = [torch.randn(2, 7, 6, 6, requires_grad=True)]
    teacher_branches = [{"scores": teacher_scores, "boxes": teacher_boxes}]
    if mechanism == "teacher_ensemble":
        teacher_branches.append(
            {
                "scores": torch.randn(2, 3, 5, requires_grad=True),
                "boxes": torch.randn(2, 4, 5, requires_grad=True),
            }
        )

    terms = plugin._compute_terms(
        student_branch={"scores": student_scores, "boxes": student_boxes},
        teacher_branches=teacher_branches,
        student_features=(student_features if spec.requires_features else None),
        teacher_features=(teacher_features if spec.requires_features else None),
    )
    terms["total"].backward()

    assert torch.allclose(terms["total"], terms[mechanism] * 0.25)
    assert all(branch["scores"].grad is None for branch in teacher_branches)
    if mechanism == "localization":
        assert student_boxes.grad is not None
        assert teacher_boxes.grad is None
    elif spec.requires_features:
        assert student_features[0].grad is not None
        assert teacher_features[0].grad is None
    else:
        assert student_scores.grad is not None


def test_runtime_rejects_stale_feature_capture_without_current_hook_call() -> None:
    spec = DISTILLATION_MECHANISMS["feature"]
    plugin = YOLO26DistillationRuntimePlugin(
        mechanism="feature",
        component_id=spec.component_id,
        changed_variable=spec.changed_variable,
    )

    with pytest.raises(ValueError, match="hooks did not fire"):
        plugin._ordered_features(
            {"model.16": torch.randn(1, 2, 2, 2)},
            {"model.16": 0},
            {},
        )


def test_mechanism_resume_state_is_scoped_and_payload_bound(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    student = tmp_path / "yolo26n.pt"
    checkpoint = tmp_path / "last.pt"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    checkpoint.write_bytes(b"checkpoint")
    spec = DISTILLATION_MECHANISMS["logits"]
    plugin = YOLO26DistillationRuntimePlugin(
        mechanism="logits",
        component_id=spec.component_id,
        changed_variable=spec.changed_variable,
        teacher=str(teacher),
        student=str(student),
    )
    context = SimpleNamespace(
        payload_path=tmp_path / "adapter_runtime_payload.yaml",
        payload=SimpleNamespace(protocol_hash="protocol-1", payload_hash="payload-1"),
    )
    state = {
        "config_hash": plugin._config_hash,
        "protocol_hash": "protocol-1",
        "teacher_checkpoint_sha256": _sha(teacher),
        "teacher_checkpoint_sha256s": [_sha(teacher)],
        "runtime_payload_hash": "other-payload",
        "component_id": spec.component_id,
        "mechanism": "logits",
    }
    sidecar = checkpoint.with_suffix(".pt.distillation.logits.json")
    sidecar.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime_payload_hash"):
        plugin.on_checkpoint_load(
            context=context,
            trainer=SimpleNamespace(args=SimpleNamespace(resume=str(checkpoint))),
            checkpoint={},
        )
    assert sidecar.is_file()


@pytest.mark.parametrize("mechanism", sorted(DISTILLATION_MECHANISMS))
def test_adapter_payload_uses_explicit_component_registry_identity(
    mechanism: str,
    tmp_path: Path,
) -> None:
    spec = DISTILLATION_MECHANISMS[mechanism]
    teacher = tmp_path / "yolo26s.pt"
    ensemble_teacher = tmp_path / "yolo26m.pt"
    student = tmp_path / "yolo26n.pt"
    for path in (teacher, ensemble_teacher, student):
        path.write_bytes(path.name.encode("ascii"))
    options = {
        "teacher": str(teacher),
        "student": str(student),
        "teacher_data": "coco.yaml",
        "student_data": "coco.yaml",
        spec.changed_variable: 0.2,
    }
    if mechanism == "teacher_ensemble":
        options["teachers"] = [str(ensemble_teacher)]
    contract = ComponentContract(
        component_id=spec.component_id,
        display_name=mechanism,
        category="distillation",
        implementation_path=(
            "yolo_agent.components.adapters.distillation.yolo26_distillation"
        ),
        adapter_class="YOLO26DistillationAdapter",
        maturity="adapter_implemented",
    )
    context = AdapterContext(
        contract=contract,
        detector_family="yolo26",
        imgsz=640,
        workspace=tmp_path,
        options=options,
    )

    payload = YOLO26DistillationAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=[
            "yolo",
            "detect",
            "train",
            f"model={student}",
            "data=coco.yaml",
            "imgsz=640",
        ],
        generated_config={},
    )

    assert payload.component_ids == [spec.component_id]
    assert payload.changed_variables == {spec.changed_variable: 0.2}
    assert payload.loss_plugin[0].options["mechanism"] == mechanism
    assert payload.expected_artifacts[0].name == f"distillation_{mechanism}_evidence"


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_relation_distillation_bounds_quadratic_spatial_matrix() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 2, 4),
        teacher_logits=torch.randn(1, 2, 4),
        student_features=[torch.randn(1, 4, 40, 40, requires_grad=True)],
        teacher_features=[torch.randn(1, 8, 40, 40)],
    )

    output = build_distillation_mechanism_loss(
        "relation", max_spatial_tokens=64
    ).compute(inputs)

    assert output.metrics["max_relation_tokens"] <= 64


def test_base_distillation_losses_reject_incompatible_inputs() -> None:
    inputs = DistillationInputs(
        student_logits=torch.randn(2, 4),
        teacher_logits=torch.randn(3, 4),
    )
    with pytest.raises(ValueError, match="identical shapes"):
        build_distillation_mechanism_loss("logits").compute(inputs)
    with pytest.raises(ValueError, match="requires student and teacher boxes"):
        build_distillation_mechanism_loss("localization").compute(inputs)
