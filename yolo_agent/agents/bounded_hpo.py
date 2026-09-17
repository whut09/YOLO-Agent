"""Bounded HPO over ActionSpec-declared search spaces only.

When an implementation-ready action is selected, the loop may tune *only*
the parameters its ``ActionSpec.search_space`` explicitly declares — loss
gamma, loss weight, sampling ratio, augmentation probability, confidence
threshold, NMS IoU, temperature, distillation weight.  Anything outside the
declared surface is rejected structurally, never swept.  Train-time HPO
shapes future ASHA assignments; inference-only actions tune through the
prediction/eval route without touching the training graph, and confidence /
NMS thresholds tune on the validation split only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.action_space_schemas import ActionSpec
from yolo_agent.agents.error_delta_profile import ErrorDeltaProfile


HpoScope = Literal["asha_train_time", "eval_route", "forbidden"]


class HpoParamSpec(BaseModel):
    """One tunable parameter as declared by an ActionSpec search space."""

    name: str
    values: list[float | int | str] = Field(min_length=1)


class BoundedSearchSpace(BaseModel):
    """The exact parameter surface an action authorizes tuning."""

    action_id: str
    params: list[HpoParamSpec] = Field(default_factory=list)

    def allows(self, name: str) -> bool:
        return any(param.name == name for param in self.params)

    def param(self, name: str) -> HpoParamSpec | None:
        return next((item for item in self.params if item.name == name), None)

    def is_empty(self) -> bool:
        return not self.params


def search_space_from_action(action: ActionSpec) -> BoundedSearchSpace:
    """Extract the bounded surface from one ActionSpec declaration."""

    params = [
        HpoParamSpec(name=name, values=list(values))
        for name, values in action.search_space.items()
        if isinstance(values, list) and values
    ]
    return BoundedSearchSpace(action_id=action.action_id, params=params)


def hpo_scope_for_action(action: ActionSpec) -> HpoScope:
    """Route HPO by the action's runtime phase — never by convenience.

    Train-time actions tune under ASHA budget assignment.  Inference-only
    actions tune through the prediction/eval route.  Threshold-family
    actions may only tune on validation data (enforced by callers through
    the split contract on eval-route trials).
    """

    if action.inference_only:
        return "eval_route"
    if action.train_time:
        return "asha_train_time"
    return "forbidden"


class HpoRequestRejected(Exception):
    """A request to tune outside the bounded surface."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = list(reasons)
        super().__init__("; ".join(reasons))


def request_hpo_points(
    action: ActionSpec,
    requested_params: dict[str, list[float | int | str]],
    *,
    split: str = "val",
) -> list[dict[str, float | int | str]]:
    """Validate an HPO request against the action's declared search space.

    Every requested parameter must exist in ``ActionSpec.search_space`` and
    every requested value must be one of the declared grid values.  Violations
    raise :class:`HpoRequestRejected` — silent pruning of out-of-scope keys is
    forbidden because it would hide contract drift.  Confidence/NMS thresholds
    are validation-split-only.
    """

    space = search_space_from_action(action)
    reasons: list[str] = []
    for name in requested_params:
        if not space.allows(name):
            reasons.append(f"param_not_in_declared_search_space:{action.action_id}:{name}")
            continue
        declared = space.param(name)
        assert declared is not None
        outside = [str(value) for value in requested_params[name] if value not in declared.values]
        if outside:
            reasons.append(
                f"values_outside_declared_grid:{action.action_id}:{name}:[{','.join(outside)}]"
            )
    if reasons:
        raise HpoRequestRejected(reasons)
    if action.family in {"threshold", "postprocess", "calibration"} and split != "val":
        raise HpoRequestRejected(
            [f"threshold_tuning_requires_validation_split:{action.action_id}:got={split}"]
        )
    return [
        dict(zip(sorted(requested_params), values, strict=False))
        for values in _grid(sorted(requested_params), requested_params)
    ]


def _grid(
    names: list[str],
    requested: dict[str, list[float | int | str]],
) -> list[tuple[float | int | str, ...]]:
    if not names:
        return [()]
    head, rest = names[0], names[1:]
    sub_grids = _grid(rest, requested)
    return [(value, *sub) for value in requested[head] for sub in sub_grids]


class EvalRouteTrial(BaseModel):
    """One inference/eval-route tuning trial — no training allocation."""

    trial_id: str
    action_id: str
    split: Literal["val"] = "val"
    params: dict[str, float | int | str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)

    def applies_to_training(self) -> bool:
        """Eval-route trials never authorize training-graph changes."""

        return False


def rank_eval_route_trials(
    trials: list[EvalRouteTrial],
    *,
    primary_metric: str = "map50_95",
    max_latency: float | None = None,
) -> list[EvalRouteTrial]:
    """Rank eval-route trials under the same hard-constraint discipline."""

    eligible = trials
    if max_latency is not None:
        eligible = [
            trial
            for trial in trials
            if trial.metrics.get("latency_ms") is None
            or float(trial.metrics["latency_ms"]) <= max_latency
        ]
    return sorted(eligible, key=lambda t: (-t.metrics.get(primary_metric, float("-inf")), t.trial_id))


def should_refine_hpo(
    profile: ErrorDeltaProfile,
    space: BoundedSearchSpace,
    *,
    tried_points: int,
    max_points: int,
) -> bool:
    """Whether the bounded surface still has untried budget worth sweeping."""

    if tried_points >= max_points:
        return False
    if space.is_empty():
        return False
    return profile.primary_effect_delta() is None or profile.primary_effect_delta() <= 0


__all__ = [
    "BoundedSearchSpace",
    "EvalRouteTrial",
    "HpoParamSpec",
    "HpoRequestRejected",
    "HpoScope",
    "hpo_scope_for_action",
    "rank_eval_route_trials",
    "request_hpo_points",
    "search_space_from_action",
    "should_refine_hpo",
]
