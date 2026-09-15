"""Training-control primitives: schedules, hooks, freeze, and parameter groups.

These are shared *primitives* only.  A paper becomes training-control ready
through its own frozen evidence (an explicit ``training_schedule`` insertion
point or an optimizer/regularization contract), its own semantic adaptation
record, and real synthetic-step probes — never because a runtime parameter
shares a name with a paper term.

No function here starts an epoch run.  Everything operates on synthetic
parameters, a single ``optimizer.step()`` / ``scheduler.step()`` at most, or
module ``requires_grad`` flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin

SchedulePolicy = Literal["constant", "linear_ramp", "step"]

RuntimePhase = Literal["pre_training", "batch_start", "batch_end", "post_training"]


class WeightSchedule(BaseModel, YAMLModelMixin):
    """Batch-indexed adaptation-weight schedule (batch-dependent policy).

    The schedule maps a batch index to the effective auxiliary-loss weight.
    ``constant`` keeps ``start_weight`` and forbids a different ``end_weight``
    so the shipped default can never silently drift from the runtime baseline.
    """

    model_config = ConfigDict(extra="forbid")

    schedule_id: str = "adaptation_weight.v1"
    policy: SchedulePolicy = "constant"
    start_weight: float = Field(ge=0.0)
    end_weight: float = Field(ge=0.0)
    warmup_batches: int = Field(default=0, ge=0)
    total_batches: int = Field(default=1, ge=1)
    step_batch: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_schedule(self) -> "WeightSchedule":
        if self.policy == "constant" and self.start_weight != self.end_weight:
            raise ValueError(
                "constant policy requires start_weight == end_weight; use "
                "linear_ramp or step for a changing weight"
            )
        if self.warmup_batches > self.total_batches:
            raise ValueError("warmup_batches must not exceed total_batches")
        if self.step_batch > self.total_batches:
            raise ValueError("step_batch must not exceed total_batches")
        if self.policy == "linear_ramp" and self.warmup_batches == 0:
            raise ValueError("linear_ramp requires warmup_batches > 0")
        return self

    def weight_at(self, batch_index: int) -> float:
        """Effective weight for a zero-based batch index (clamped at the end)."""

        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")
        if self.policy == "constant":
            return self.start_weight
        index = min(batch_index, self.total_batches)
        if self.policy == "step":
            return self.start_weight if index < self.step_batch else self.end_weight
        if index >= self.warmup_batches:
            return self.end_weight
        progress = index / self.warmup_batches
        return self.start_weight + (self.end_weight - self.start_weight) * progress


class TrainingScheduleController:
    """Runtime hook lifecycle for one weight schedule.

    ``on_train_batch_start`` announces the effective weight for the batch,
    ``on_train_batch_end`` records evidence, and ``rollback`` restores the
    pre-schedule baseline weight.  The controller never touches the model.
    """

    def __init__(self, schedule: WeightSchedule, *, baseline_weight: float) -> None:
        self.schedule = schedule
        self.baseline_weight = float(baseline_weight)
        self._batch_counter = 0
        self.announced: list[float] = []
        self.evidence: dict[str, Any] = {}

    def resolve_batch_index(self, *, batch_index: int | None, trainer: Any) -> int:
        if batch_index is not None:
            return int(batch_index)
        for attribute in ("batch_i", "batch_index"):
            value = getattr(trainer, attribute, None)
            if value is not None:
                return int(value)
        index = self._batch_counter
        self._batch_counter += 1
        return index

    def on_train_batch_start(
        self,
        *,
        batch_index: int | None = None,
        trainer: Any = None,
    ) -> float:
        index = self.resolve_batch_index(batch_index=batch_index, trainer=trainer)
        weight = self.schedule.weight_at(index)
        self.announced.append(weight)
        self.evidence["last_batch_index"] = index
        self.evidence["last_effective_weight"] = weight
        self.evidence["schedule_applied"] = True
        return weight

    def on_train_batch_end(self, *, loss_term: float | None = None) -> None:
        if loss_term is not None:
            self.evidence["last_weighted_loss_term"] = float(loss_term)

    def rollback(self) -> dict[str, float]:
        """Restore the baseline weight and return the rollback values."""

        self.announced.clear()
        self.evidence["schedule_applied"] = False
        self.evidence["last_effective_weight"] = self.baseline_weight
        return {"weight": self.baseline_weight}


class TrainingControlSpec(BaseModel, YAMLModelMixin):
    """Protocol spec for one training-control profile.

    Records defaults, paper-recommended values (only what frozen evidence
    supports), searchable parameters with valid ranges, coupled parameters,
    machine-checked invalid combinations, the runtime phase the control acts
    in, and rollback values.  ``invalid_combinations`` entries are validated
    to actually fail :class:`WeightSchedule` construction (fail-closed).
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    defaults: dict[str, float] = Field(default_factory=dict)
    paper_recommended: dict[str, float] = Field(default_factory=dict)
    searchable_parameters: list[str] = Field(min_length=1)
    valid_ranges: dict[str, list[float]]
    coupled_parameters: dict[str, list[str]] = Field(default_factory=dict)
    invalid_combinations: list[dict[str, Any]] = Field(default_factory=list)
    runtime_phase: RuntimePhase
    rollback_values: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def validate_spec(self) -> "TrainingControlSpec":
        if not self.spec_id.strip():
            raise ValueError("training-control spec requires a spec_id")
        for name in self.searchable_parameters:
            if name not in self.valid_ranges:
                raise ValueError(f"searchable parameter {name!r} lacks a valid range")
            low, high = self.valid_ranges[name][0], self.valid_ranges[name][1]
            if low > high:
                raise ValueError(f"invalid range for {name!r}: {self.valid_ranges[name]}")
        for name, value in self.paper_recommended.items():
            if name not in self.valid_ranges:
                raise ValueError(f"paper-recommended {name!r} lacks a valid range")
            low, high = self.valid_ranges[name][0], self.valid_ranges[name][1]
            if not low <= value <= high:
                raise ValueError(
                    f"paper-recommended {name}={value} outside range [{low}, {high}]"
                )
        for name, value in self.rollback_values.items():
            if name not in self.defaults or self.defaults[name] != value:
                raise ValueError(f"rollback value for {name!r} must restore the default")
        for combination in self.invalid_combinations:
            try:
                WeightSchedule.model_validate(combination)
            except ValueError:
                continue
            raise ValueError(
                f"invalid_combinations entry must fail WeightSchedule validation: {combination}"
            )
        return self


def domain_alignment_weight_spec(*, default_weight: float = 0.05) -> TrainingControlSpec:
    """Protocol spec for schedule-controlled domain-alignment weights."""

    return TrainingControlSpec(
        spec_id="training_control.domain_alignment_weight_schedule",
        defaults={
            "weight": default_weight,
            "start_weight": default_weight,
            "end_weight": default_weight,
            "warmup_batches": 0,
            "total_batches": 1,
        },
        paper_recommended={},
        searchable_parameters=["weight", "start_weight", "end_weight", "warmup_batches", "total_batches"],
        valid_ranges={
            "weight": [0.0, 10.0],
            "start_weight": [0.0, 10.0],
            "end_weight": [0.0, 10.0],
            "warmup_batches": [0.0, 100000.0],
            "total_batches": [1.0, 1000000.0],
        },
        coupled_parameters={
            "start_weight": ["end_weight", "warmup_batches"],
            "end_weight": ["start_weight", "warmup_batches"],
            "warmup_batches": ["total_batches"],
        },
        invalid_combinations=[
            {
                "policy": "constant",
                "start_weight": 0.1,
                "end_weight": 0.2,
                "warmup_batches": 0,
                "total_batches": 1,
            },
            {
                "policy": "linear_ramp",
                "start_weight": 0.0,
                "end_weight": 0.1,
                "warmup_batches": 4,
                "total_batches": 2,
            },
        ],
        runtime_phase="batch_start",
        rollback_values={
            "weight": default_weight,
            "start_weight": default_weight,
            "end_weight": default_weight,
            "warmup_batches": 0,
            "total_batches": 1,
        },
        notes=(
            "Shipped default is the constant identity schedule: zero drift "
            "from the un-scheduled runtime baseline.  Ramp/step policies are "
            "searchable until paper evidence fixes a curve."
        ),
    )


def build_parameter_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
    no_decay_substrings: tuple[str, ...] = ("bn", "bias", "norm"),
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups by name substring."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        lowered = name.lower()
        if any(token in lowered for token in no_decay_substrings):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def freeze_parameters_by_prefix(module: torch.nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    """Freeze parameters whose name starts with any prefix; return frozen names."""

    frozen: list[str] = []
    for name, parameter in module.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def unfreeze_parameters_by_prefix(module: torch.nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    """Unfreeze parameters whose name starts with any prefix; return unfrozen names."""

    unfrozen: list[str] = []
    for name, parameter in module.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
            unfrozen.append(name)
    return unfrozen


def write_schedule_yaml(schedule: WeightSchedule, path: Path | str) -> Path:
    return schedule.to_yaml(path)


__all__ = [
    "RuntimePhase",
    "SchedulePolicy",
    "TrainingControlSpec",
    "TrainingScheduleController",
    "WeightSchedule",
    "build_parameter_groups",
    "domain_alignment_weight_spec",
    "freeze_parameters_by_prefix",
    "unfreeze_parameters_by_prefix",
    "write_schedule_yaml",
]
