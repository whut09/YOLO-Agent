"""History-aware diversity, cooldown, and bounded-search policy."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.strategy_policy import CandidatePolicy


DiversityBucket = Literal["exploration", "exploitation"]
DiversityDecisionStatus = Literal["eligible", "deferred", "exhausted"]


class ExplorationDiversityPolicy(BaseModel):
    """Search diversity and stopping controls shared across auto rounds."""

    schema_version: str = "exploration_diversity.v1"
    component_family_cooldown_rounds: int = Field(default=2, ge=0)
    minimum_semantic_distance: float = Field(default=0.15, ge=0.0, le=1.0)
    no_improvement_patience: int = Field(default=5, ge=1)
    family_exhaustion_attempts: int = Field(default=4, ge=1)
    minimum_improvement: float = Field(default=0.0005, ge=0.0)
    minimum_families_for_exhaustion_stop: int = Field(default=2, ge=1)


class RecipeDescriptor(BaseModel):
    """Normalized recipe identity independent of temporary policy ids."""

    fingerprint: str
    component_family: str
    changed_values: dict[str, Any] = Field(default_factory=dict)
    semantic_tokens: list[str] = Field(default_factory=list)


class ExplorationHistoryEntry(BaseModel):
    """One executed recipe outcome used by later rounds."""

    schema_version: str = "exploration_history.v1"
    run_id: str
    round_index: int = Field(ge=1)
    policy_id: str
    candidate_id: str
    recipe_fingerprint: str
    component_family: str
    changed_values: dict[str, Any] = Field(default_factory=dict)
    semantic_tokens: list[str] = Field(default_factory=list)
    bucket: DiversityBucket
    effect_delta: float | None = None
    improved: bool = False
    completed: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DiversityDecision(BaseModel):
    policy_id: str
    status: DiversityDecisionStatus
    bucket: DiversityBucket
    recipe_fingerprint: str
    component_family: str
    nearest_semantic_distance: float | None = None
    reason: str


class DiversityStopDecision(BaseModel):
    should_stop: bool = False
    reason: str = ""
    no_improvement_rounds: int = 0
    exhausted_families: list[str] = Field(default_factory=list)


class ExplorationHistoryStore:
    """Append-only base-run history for replayable diversity decisions."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def read(self) -> list[ExplorationHistoryEntry]:
        if not self.path.is_file():
            return []
        records: list[ExplorationHistoryEntry] = []
        with self.path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                if line.strip():
                    records.append(ExplorationHistoryEntry.model_validate_json(line))
        return records

    def append(self, entries: list[ExplorationHistoryEntry]) -> list[ExplorationHistoryEntry]:
        existing = {(item.run_id, item.policy_id, item.recipe_fingerprint) for item in self.read()}
        additions = [
            item for item in entries
            if (item.run_id, item.policy_id, item.recipe_fingerprint) not in existing
        ]
        if additions:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                for item in additions:
                    file.write(item.model_dump_json() + "\n")
        return additions


def describe_recipe(proposal: CandidatePolicy) -> RecipeDescriptor:
    """Build an exact fingerprint plus a coarser semantic representation."""
    changed = {
        str(key): _normalize_value(value)
        for key, value in proposal.train_overrides.items()
        if str(key) != "imgsz"
    }
    payload = {
        "action_domain": proposal.action_domain,
        "components": sorted(proposal.components),
        "train_overrides": changed,
        "execution_action": proposal.execution_action,
        "base_model": proposal.base_model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    family = _component_family(proposal, changed)
    tokens = sorted(
        {
            proposal.action_domain,
            family,
            *proposal.components,
            *changed.keys(),
            *(_normalize_action_id(proposal.action_id).split("_") if proposal.action_id else []),
        }
    )
    return RecipeDescriptor(
        fingerprint=hashlib.sha256(encoded).hexdigest(),
        component_family=family,
        changed_values=changed,
        semantic_tokens=tokens,
    )


def screen_proposal(
    proposal: CandidatePolicy,
    *,
    current_round: int,
    history: list[ExplorationHistoryEntry],
    policy: ExplorationDiversityPolicy,
) -> DiversityDecision:
    """Apply exact dedupe, family exhaustion, cooldown, and semantic distance."""
    descriptor = describe_recipe(proposal)
    family_history = [item for item in history if item.component_family == descriptor.component_family]
    bucket: DiversityBucket = (
        "exploitation"
        if any(item.improved and (item.effect_delta or 0.0) > policy.minimum_improvement for item in family_history)
        else "exploration"
    )
    if any(item.recipe_fingerprint == descriptor.fingerprint for item in history):
        return _decision(proposal, descriptor, bucket, "deferred", 0.0, "duplicate_recipe_fingerprint")
    if _family_exhausted(family_history, policy):
        return _decision(proposal, descriptor, bucket, "exhausted", None, "component_family_exhausted")
    if family_history:
        latest_round = max(item.round_index for item in family_history)
        if current_round - latest_round <= policy.component_family_cooldown_rounds:
            return _decision(
                proposal, descriptor, bucket, "deferred", None,
                f"component_family_cooldown:{descriptor.component_family}:last_round={latest_round}",
            )
    distances = [semantic_distance(descriptor, _descriptor_from_history(item)) for item in history]
    nearest = min(distances) if distances else None
    if nearest is not None and nearest < policy.minimum_semantic_distance:
        return _decision(
            proposal, descriptor, bucket, "deferred", nearest,
            f"minimum_semantic_distance:{nearest:.6f}<{policy.minimum_semantic_distance:.6f}",
        )
    return _decision(proposal, descriptor, bucket, "eligible", nearest, "diversity_guards_passed")


def semantic_distance(first: RecipeDescriptor, second: RecipeDescriptor) -> float:
    """Return 0 for equivalent nearby recipes and 1 for distinct families."""
    if first.fingerprint == second.fingerprint:
        return 0.0
    if first.component_family != second.component_family:
        return 1.0
    keys = set(first.changed_values) | set(second.changed_values)
    if keys:
        distances = [
            _value_distance(first.changed_values.get(key), second.changed_values.get(key))
            for key in keys
        ]
        return min(1.0, sum(distances) / len(distances))
    first_tokens, second_tokens = set(first.semantic_tokens), set(second.semantic_tokens)
    union = first_tokens | second_tokens
    return 1.0 - (len(first_tokens & second_tokens) / len(union)) if union else 0.0


def evaluate_diversity_stop(
    history: list[ExplorationHistoryEntry],
    policy: ExplorationDiversityPolicy,
) -> DiversityStopDecision:
    """Stop after stagnant rounds or when every attempted family is exhausted."""
    completed = [item for item in history if item.completed]
    rounds: dict[int, list[ExplorationHistoryEntry]] = {}
    for item in completed:
        rounds.setdefault(item.round_index, []).append(item)
    stagnant = 0
    for round_index in sorted(rounds, reverse=True):
        if any(item.improved and (item.effect_delta or 0.0) > policy.minimum_improvement for item in rounds[round_index]):
            break
        stagnant += 1
    if stagnant >= policy.no_improvement_patience:
        return DiversityStopDecision(
            should_stop=True, reason="no_improvement_patience", no_improvement_rounds=stagnant,
            exhausted_families=exhausted_families(completed, policy),
        )
    families = sorted({item.component_family for item in completed})
    exhausted = exhausted_families(completed, policy)
    if len(families) >= policy.minimum_families_for_exhaustion_stop and exhausted == families:
        return DiversityStopDecision(
            should_stop=True, reason="family_exhaustion", no_improvement_rounds=stagnant,
            exhausted_families=exhausted,
        )
    return DiversityStopDecision(
        no_improvement_rounds=stagnant, exhausted_families=exhausted,
    )


def exhausted_families(
    history: list[ExplorationHistoryEntry], policy: ExplorationDiversityPolicy,
) -> list[str]:
    families = sorted({item.component_family for item in history})
    return [
        family for family in families
        if _family_exhausted([item for item in history if item.component_family == family], policy)
    ]


def _decision(
    proposal: CandidatePolicy,
    descriptor: RecipeDescriptor,
    bucket: DiversityBucket,
    status: DiversityDecisionStatus,
    distance: float | None,
    reason: str,
) -> DiversityDecision:
    return DiversityDecision(
        policy_id=proposal.policy_id,
        status=status,
        bucket=bucket,
        recipe_fingerprint=descriptor.fingerprint,
        component_family=descriptor.component_family,
        nearest_semantic_distance=distance,
        reason=reason,
    )


def _descriptor_from_history(item: ExplorationHistoryEntry) -> RecipeDescriptor:
    return RecipeDescriptor(
        fingerprint=item.recipe_fingerprint,
        component_family=item.component_family,
        changed_values=item.changed_values,
        semantic_tokens=item.semantic_tokens,
    )


def _family_exhausted(
    history: list[ExplorationHistoryEntry], policy: ExplorationDiversityPolicy,
) -> bool:
    completed = [item for item in history if item.completed]
    return (
        len(completed) >= policy.family_exhaustion_attempts
        and not any(
            item.improved and (item.effect_delta or 0.0) > policy.minimum_improvement
            for item in completed
        )
    )


def _component_family(proposal: CandidatePolicy, changed: dict[str, Any]) -> str:
    keys = set(changed)
    if keys & {"mosaic", "mixup", "copy_paste", "close_mosaic", "degrees", "translate", "scale"}:
        return "augmentation:" + sorted(keys & {
            "mosaic", "mixup", "copy_paste", "close_mosaic", "degrees", "translate", "scale"
        })[0]
    if keys & {"box", "cls", "dfl"}:
        return "loss:" + sorted(keys & {"box", "cls", "dfl"})[0]
    if keys & {"lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"}:
        return "optimizer_schedule"
    if proposal.components:
        parts = proposal.components[0].split(".")
        return "component:" + ".".join(parts[:2])
    normalized_action = _normalize_action_id(proposal.action_id)
    if normalized_action:
        return f"{proposal.action_domain}:{normalized_action}"
    return proposal.action_domain


def _normalize_action_id(action_id: str | None) -> str:
    text = str(action_id or "").strip().lower()
    text = re.sub(r"(?:_|^)(?:-?\d+(?:[._]\d+)?)+$", "", text)
    return text.strip("_")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _value_distance(first: Any, second: Any) -> float:
    if first is None or second is None:
        return 1.0
    if (
        isinstance(first, (int, float))
        and not isinstance(first, bool)
        and isinstance(second, (int, float))
        and not isinstance(second, bool)
    ):
        return min(1.0, abs(float(first) - float(second)) / max(abs(float(first)), abs(float(second)), 1.0))
    return 0.0 if first == second else 1.0
