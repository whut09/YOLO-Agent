"""Learn reusable policy effects from error-fact deltas."""

from __future__ import annotations

from typing import Any

from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue
from yolo_agent.core.matched_baseline import paired_metric_delta
from yolo_agent.core.policy_memory import (
    ActionFingerprint,
    PolicyActionCost,
    PolicyFidelity,
    PolicyMemoryRecord,
    PolicyMemoryStore,
)


class PolicyLearner:
    """Convert parent/current error deltas into append-only policy memory."""

    def __init__(self, memory_store: PolicyMemoryStore | None = None) -> None:
        self.memory_store = memory_store or PolicyMemoryStore()

    def learn_from_error_delta(
        self,
        run_id: str,
        parent_run_id: str | None,
        dataset_version: str,
        error_delta: dict[str, Any],
        current_evidence: Evidence | None = None,
        parent_evidence: Evidence | None = None,
        changed_variables: dict[str, Any] | None = None,
        scenario: str | None = None,
        recipe_id: str | None = None,
        component_versions: dict[str, str] | None = None,
        model_family: str = "unknown",
        dataset_signature: str | None = None,
        protocol_hash: str = "unknown",
        fidelity: PolicyFidelity = "unknown",
        action_before_values: dict[str, Any] | None = None,
        append: bool = True,
    ) -> list[PolicyMemoryRecord]:
        """Create policy-memory records from comparable error delta rows."""
        records: list[PolicyMemoryRecord] = []
        changes = changed_variables or {}
        actual_actions = actions_from_changed_variables(changes)
        for item in _delta_items(error_delta):
            if not isinstance(item, dict):
                continue
            before = _numeric(item.get("parent_value"))
            after = _numeric(item.get("current_value"))
            delta = _numeric(item.get("delta"))
            matched_control_hash = _optional_str(item.get("matched_control_hash"))
            if before is None or after is None or delta is None or matched_control_hash is None:
                continue
            actions, inferred = _actions_for_item(item, actual_actions)
            if not actions:
                continue
            higher_is_better = _higher_is_better(item)
            cost = _cost_for_item(item, current_evidence=current_evidence, parent_evidence=parent_evidence)
            seed_count = _paired_seed_count(
                current_evidence,
                str(item.get("candidate_id") or ""),
                str(item.get("metric_name") or ""),
            )
            confidence, reason = _confidence(seed_count, inferred=inferred, changed_variables=changes)
            for action in actions:
                changed_variable, after_value = action_transition(action, changes)
                records.append(
                    PolicyMemoryRecord(
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        dataset_version=dataset_version,
                        scenario=scenario,
                        action=action,
                        action_fingerprint=ActionFingerprint(
                            action=action,
                            recipe_id=recipe_id,
                            component_versions=component_versions or {},
                            changed_variable=changed_variable,
                            before_value=(action_before_values or {}).get(changed_variable),
                            after_value=after_value,
                            model_family=model_family,
                            dataset_signature=dataset_signature or dataset_version,
                            protocol_hash=protocol_hash,
                            fidelity=fidelity,
                            matched_control_hash=matched_control_hash,
                        ),
                        target=target_from_delta_item(item),
                        target_fact_type=_optional_str(item.get("fact_type")),
                        target_subject=_optional_str(item.get("subject")),
                        class_name=_optional_str(item.get("class_name")),
                        class_pair=_optional_str(item.get("class_pair")),
                        area=_optional_str(item.get("area")),
                        metric_name=_optional_str(item.get("metric_name")),
                        before=before,
                        after=after,
                        delta=delta,
                        higher_is_better=higher_is_better,
                        trend=_trend(item),
                        candidate_id=_optional_str(item.get("candidate_id")),
                        node_id=_optional_str(item.get("node_id")),
                        cost=cost,
                        confidence=confidence,
                        confidence_reason=reason,
                        seed_count=seed_count,
                        changed_variables=changes,
                        inferred_action=inferred,
                        matched_control_hash=matched_control_hash,
                    )
                )
        if append:
            return self.memory_store.append(records)
        return records


def actions_from_changed_variables(changed_variables: dict[str, Any]) -> list[str]:
    """Flatten changed variables into stable action identifiers."""
    actions: list[str] = []
    for key, value in sorted(changed_variables.items()):
        values = value if isinstance(value, list) else [value]
        for raw in values:
            if raw is None:
                continue
            text = str(raw)
            if text:
                actions.append(text)
        if not values:
            actions.append(str(key))
    return list(dict.fromkeys(actions))


def action_transition(action: str, changed_variables: dict[str, Any]) -> tuple[str, Any]:
    """Return the changed variable and value represented by one action."""
    for key, value in sorted(changed_variables.items()):
        values = value if isinstance(value, list) else [value]
        if action in {str(item) for item in values if item is not None}:
            return str(key), value
    return "inferred_action", action


def target_from_delta_item(item: dict[str, Any]) -> str:
    """Return a stable learning target for one error-delta item."""
    fact_type = str(item.get("fact_type") or "error_fact")
    subject = str(item.get("class_name") or item.get("class_pair") or item.get("area") or item.get("subject") or "unknown")
    metric_name = str(item.get("metric_name") or "count")
    return f"{fact_type}:{subject}:{metric_name}"


def _delta_items(error_delta: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key in ("improved_errors", "regressed_errors", "unchanged_errors"):
        value = error_delta.get(key, [])
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))
    return items


def _actions_for_item(item: dict[str, Any], actual_actions: list[str]) -> tuple[list[str], bool]:
    if actual_actions:
        return actual_actions, False
    raw = item.get("action_candidates", [])
    if not isinstance(raw, list):
        return [], True
    return list(dict.fromkeys(str(action) for action in raw if action is not None)), True


def _cost_for_item(
    item: dict[str, Any],
    current_evidence: Evidence | None,
    parent_evidence: Evidence | None,
) -> PolicyActionCost:
    candidate_id = _optional_str(item.get("candidate_id"))
    node_id = _optional_str(item.get("node_id"))
    latency_before, latency_after = _paired_cost_values(
        current_evidence, candidate_id, node_id, "latency_ms"
    )
    size_before, size_after = _paired_cost_values(
        current_evidence, candidate_id, node_id, "model_size_mb"
    )
    current_records = current_evidence.metric_records if current_evidence is not None else []
    current_index = EvidenceIndex(current_records)
    gpu_hours = _metric(current_index, "gpu_hours", candidate_id=candidate_id, node_id=node_id)
    return PolicyActionCost(
        latency_before_ms=latency_before,
        latency_after_ms=latency_after,
        latency_delta_ms=_delta(latency_before, latency_after),
        latency_delta_pct=_delta_pct(latency_before, latency_after),
        model_size_before_mb=size_before,
        model_size_after_mb=size_after,
        model_size_delta_mb=_delta(size_before, size_after),
        model_size_delta_pct=_delta_pct(size_before, size_after),
        gpu_hours=gpu_hours,
    )


def _paired_cost_values(
    evidence: Evidence | None,
    candidate_id: str | None,
    node_id: str | None,
    metric_name: str,
) -> tuple[float | None, float | None]:
    if evidence is None or not candidate_id or not node_id:
        return None, None
    matches = [
        record
        for record in evidence.metric_records
        if record.candidate_id == candidate_id
        and record.node_id == node_id
        and record.metric_name == metric_name
        and record.evidence_role == "current_observation"
        and record.inheritance_depth == 0
    ]
    if not matches:
        return None, None
    candidate = max(matches, key=lambda record: record.created_at)
    _, delta = paired_metric_delta(candidate, evidence.metric_records)
    if delta is None:
        return None, None
    return delta.baseline_value, delta.candidate_value


def _metric(index: EvidenceIndex, metric_name: str, candidate_id: str | None = None, node_id: str | None = None) -> float | None:
    filters: dict[str, object] = {"metric_name": metric_name, "verified": True}
    if candidate_id:
        filters["candidate_id"] = candidate_id
    if node_id:
        filters["node_id"] = node_id
    value = index.metric_value(**filters)
    return _numeric(value)


def _seed_count(evidence: Evidence | None, candidate_id: str, metric_name: str) -> int:
    if evidence is None or not candidate_id or not metric_name:
        return 1
    records = [
        record
        for record in evidence.metric_records
        if record.candidate_id == candidate_id and record.metric_name == metric_name and record.verified
    ]
    seeds = {_seed_from_record(record) for record in records}
    seeds.discard(None)
    if seeds:
        return len(seeds)
    return max(1, len({record.node_id for record in records}) or len(records))


def _paired_seed_count(evidence: Evidence | None, candidate_id: str, metric_name: str) -> int:
    """Count only seeds that have a verified exact matched-control pair."""
    if evidence is None or not candidate_id or not metric_name:
        return 1
    seeds: set[str] = set()
    for record in evidence.metric_records:
        if (
            record.candidate_id != candidate_id
            or record.metric_name != metric_name
            or record.evidence_role != "current_observation"
            or record.inheritance_depth > 0
        ):
            continue
        _, delta = paired_metric_delta(record, evidence.metric_records)
        if delta is not None:
            seeds.add(delta.match_key.seed)
    return max(1, len(seeds))


def _seed_from_record(record: MetricEvidence) -> str | None:
    node_id = record.node_id.lower()
    for marker in ("seed", "s"):
        index = node_id.find(marker)
        if index >= 0:
            suffix = node_id[index + len(marker):].lstrip("_-")
            digits = "".join(char for char in suffix if char.isdigit())
            if digits:
                return digits
    return None


def _confidence(seed_count: int, inferred: bool, changed_variables: dict[str, Any]) -> tuple[str, str]:
    if inferred:
        return "low", "action inferred from error-fact candidates, not an executed variable"
    if seed_count >= 3:
        return "high", "executed variable with three or more verified seeds"
    if seed_count >= 2:
        return "medium", "executed variable with repeated verified seeds"
    if changed_variables:
        return "low", "executed variable with a single verified seed"
    return "low", "single observation"


def _higher_is_better(item: dict[str, Any]) -> bool:
    fact_type = str(item.get("fact_type") or "")
    if fact_type in {
        "false_negative_heavy_class",
        "localization_heavy_class",
        "class_confusion_pair",
        "background_false_positive_class",
    }:
        return False
    return True


def _trend(item: dict[str, Any]) -> str:
    trend = str(item.get("trend") or "unchanged")
    if trend in {"improved", "regressed", "unchanged", "resolved", "new", "current"}:
        return trend
    return "unchanged"


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return round(after - before, 6)


def _delta_pct(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return round((after - before) / before * 100.0, 6)


def _numeric(value: MetricValue | object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
