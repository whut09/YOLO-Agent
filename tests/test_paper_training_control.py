"""Training-control audit tests: schedules, hooks, freeze, rollback, scope.

No test here runs epoch training or creates an ExperimentNode.  Every probe
uses synthetic parameters and at most one synthetic optimizer.step() per
scheduled batch index.
"""

from __future__ import annotations

from pathlib import Path

import pydantic
import pytest
import torch

from yolo_agent.components.training_control import (
    TrainingControlSpec,
    TrainingScheduleController,
    WeightSchedule,
    build_parameter_groups,
    domain_alignment_weight_spec,
    freeze_parameters_by_prefix,
    unfreeze_parameters_by_prefix,
)
from yolo_agent.research.paper_training_control import (
    PaperTrainingControlAuditBuilder,
    summarize_training_control_audit,
)
from yolo_agent.research.training_control_adaptation import (
    RUNTIME_BINDING_POINTS,
    TrainingParameterAdaptation,
)


def test_schedule_math_is_exact_and_clamped() -> None:
    ramp = WeightSchedule(
        policy="linear_ramp",
        start_weight=0.0,
        end_weight=0.1,
        warmup_batches=4,
        total_batches=8,
    )
    assert [round(ramp.weight_at(i), 4) for i in range(4)] == [0.0, 0.025, 0.05, 0.075]
    assert ramp.weight_at(4) == 0.1
    # Beyond the horizon the schedule holds its final value.
    assert ramp.weight_at(999) == 0.1

    stepped = WeightSchedule(
        policy="step",
        start_weight=0.2,
        end_weight=0.05,
        step_batch=3,
        total_batches=8,
    )
    assert stepped.weight_at(2) == 0.2
    assert stepped.weight_at(3) == 0.05

    constant = WeightSchedule(start_weight=0.05, end_weight=0.05, total_batches=4)
    assert all(constant.weight_at(i) == 0.05 for i in range(4))

    with pytest.raises(ValueError, match="non-negative"):
        ramp.weight_at(-1)


def test_invalid_schedule_combinations_fail_closed() -> None:
    with pytest.raises(pydantic.ValidationError, match="constant"):
        WeightSchedule(start_weight=0.1, end_weight=0.2, total_batches=4)
    with pytest.raises(pydantic.ValidationError, match="warmup_batches"):
        WeightSchedule(
            policy="linear_ramp",
            start_weight=0.0,
            end_weight=0.1,
            warmup_batches=5,
            total_batches=4,
        )


def test_schedule_yaml_config_round_trip() -> None:
    schedule = WeightSchedule(
        policy="linear_ramp",
        start_weight=0.0,
        end_weight=0.1,
        warmup_batches=4,
        total_batches=8,
    )
    path = Path(".tmp_schedule_roundtrip.yaml")
    schedule.to_yaml(path)
    restored = WeightSchedule.from_yaml(path)
    assert restored == schedule
    path.unlink()


def test_controller_hook_lifecycle_and_rollback() -> None:
    schedule = WeightSchedule(
        policy="linear_ramp",
        start_weight=0.0,
        end_weight=0.1,
        warmup_batches=4,
        total_batches=8,
    )
    controller = TrainingScheduleController(schedule, baseline_weight=0.05)
    assert controller.on_train_batch_start(batch_index=0, trainer=None) == 0.0
    assert controller.on_train_batch_start(batch_index=8, trainer=None) == 0.1
    controller.on_train_batch_end(loss_term=0.5)
    assert controller.evidence["last_weighted_loss_term"] == 0.5
    assert controller.evidence["schedule_applied"] is True

    rollback = controller.rollback()
    assert rollback == {"weight": 0.05}
    assert controller.evidence["schedule_applied"] is False
    assert controller.evidence["last_effective_weight"] == 0.05
    assert controller.announced == []


def test_identity_schedule_never_drifts_from_baseline() -> None:
    schedule = WeightSchedule(
        policy="constant",
        start_weight=0.05,
        end_weight=0.05,
        total_batches=4,
    )
    controller = TrainingScheduleController(schedule, baseline_weight=0.05)
    for index in range(4):
        assert controller.on_train_batch_start(batch_index=index, trainer=None) == 0.05


def test_synthetic_single_optimizer_step_with_parameter_groups() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Linear(4, 2),
    )
    groups = build_parameter_groups(model, weight_decay=5e-4)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 5e-4
    assert groups[1]["weight_decay"] == 0.0
    optimizer = torch.optim.SGD(groups, lr=0.1)
    loss = model(torch.randn(8, 4)).square().sum()
    loss.backward()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.step()
    changed = any(
        not torch.equal(before_parameter, parameter.detach())
        for before_parameter, parameter in zip(before, model.parameters())
    )
    assert changed


def test_freeze_unfreeze_round_trip() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    frozen = freeze_parameters_by_prefix(model, ("0.",))
    assert frozen
    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad
    unfrozen = unfreeze_parameters_by_prefix(model, ("0.",))
    assert unfrozen == frozen
    assert model[0].weight.requires_grad


def test_protocol_spec_validates_ranges_rollback_and_invalid_combinations() -> None:
    spec = domain_alignment_weight_spec()
    assert spec.searchable_parameters
    assert spec.coupled_parameters
    assert spec.invalid_combinations
    # Every declared invalid combination must actually fail schedule
    # construction (fail-closed): a typo'd "invalid" combo is itself invalid.
    for combination in spec.invalid_combinations:
        with pytest.raises(pydantic.ValidationError):
            WeightSchedule.model_validate(combination)
    # Rollback values restore defaults exactly.
    for name, value in spec.rollback_values.items():
        assert spec.defaults[name] == value
    with pytest.raises(pydantic.ValidationError, match="range"):
        TrainingControlSpec.model_validate(
            spec.model_copy(update={"paper_recommended": {"warmup_batches": 200000.0}}).model_dump()
        )
    with pytest.raises(pydantic.ValidationError, match="rollback"):
        TrainingControlSpec.model_validate(
            spec.model_copy(update={"rollback_values": {"weight": 0.9}}).model_dump()
        )


def test_semantic_adaptation_rejects_nonexistent_runtime_bindings() -> None:
    real = TrainingParameterAdaptation(
        paper_id="ecva:eccv2024:7083",
        parameter_name="alignment_loss_weight",
        paper_semantics="batch-dependent auxiliary weight",
        runtime_contract="WeightSchedule × strategy loss",
        runtime_binding_point="compute_loss",
        ultralytics_hook="trainer compute_loss stage",
        preserved_information=["weighting semantics"],
        evidence_refs=["frozen plan"],
    )
    assert real.runtime_binding_point in RUNTIME_BINDING_POINTS
    with pytest.raises(pydantic.ValidationError, match="does not exist"):
        TrainingParameterAdaptation.model_validate(
            real.model_copy(update={"runtime_binding_point": "lr0"}).model_dump()
        )
    with pytest.raises(pydantic.ValidationError, match="exact_reproduction"):
        TrainingParameterAdaptation.model_validate(
            real.model_copy(
                update={
                    "adaptation_class": "exact_reproduction",
                    "approximated_information": ["curve"],
                }
            ).model_dump()
        )


def test_audit_scope_comes_from_frozen_evidence_not_domain_names() -> None:
    records = PaperTrainingControlAuditBuilder().build()
    assert len(records) == 83
    summary = summarize_training_control_audit(records)
    # Exactly one paper's frozen evidence declares a training_schedule
    # insertion point; no paper carries a training-control domain.
    assert summary["in_scope"] == 1
    assert summary["ready"] == 1
    assert summary["blocked_missing_evidence"] == 0
    assert summary["out_of_scope"] == 82

    by_id = {record.paper_id: record for record in records}
    scheduled = by_id["ecva:eccv2024:7083"]
    assert scheduled.in_scope
    assert scheduled.status == "ready"
    assert scheduled.scope_reason == "frozen plan declares a training_schedule insertion point"
    assert scheduled.adaptations
    assert scheduled.behavior_passed
    assert scheduled.protocol_spec_ids

    out_of_scope = by_id["arxiv:2103.14259"]
    assert not out_of_scope.in_scope
    assert out_of_scope.status == "out_of_scope"
    assert "training_schedule" in out_of_scope.scope_reason


def test_ready_record_semantics_bind_paper_to_runtime_not_names() -> None:
    records = PaperTrainingControlAuditBuilder().build()
    scheduled = next(record for record in records if record.in_scope)
    adaptation = scheduled.adaptations[0]
    assert adaptation.parameter_name == "alignment_loss_weight"
    assert adaptation.runtime_binding_point in RUNTIME_BINDING_POINTS
    assert adaptation.adaptation_class == "faithful_adaptation"
    assert adaptation.preserved_information
    # Curve values are not certified by frozen evidence: recorded honestly as
    # an approximation, and the curve stays searchable in the ProtocolSpec.
    assert adaptation.approximated_information
