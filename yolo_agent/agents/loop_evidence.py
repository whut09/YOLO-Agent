"""Evidence gates, lineage snapshots, and next-round evidence deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_agent.agents.loop_io import read_json, read_yaml, write_json
from yolo_agent.core.evidence_contract import EvidenceGate, EvidenceGateResult, default_loop_evidence_requirements
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue
from yolo_agent.core.loop_state import LoopState
from yolo_agent.core.run_context import RunContext
from yolo_agent.core.run_lineage import RunLineageStore, build_lineage_record


class LoopEvidence:
    """Own evidence-gate evaluation and lineage recording for one loop run."""

    def __init__(
        self,
        context: RunContext,
        state: LoopState,
        evidence_store: EvidenceStore,
        lineage_store: RunLineageStore,
    ) -> None:
        self.context = context
        self.state = state
        self.evidence_store = evidence_store
        self.lineage_store = lineage_store

    def current_gate(self) -> EvidenceGateResult:
        """Evaluate the current evidence gate."""
        evidence = self.evidence_store.load_run(self.context.run_id)
        extra = loop_plan_evidence_required(self.context.artifact_path("loop_plan.yaml"))
        return EvidenceGate(default_loop_evidence_requirements(extra)).evaluate(
            evidence=evidence,
            artifacts=self.state.artifacts,
        )

    def write_status(self) -> Path:
        """Write evidence gate status and append a lineage snapshot."""
        gate = self.current_gate()
        path = self.context.artifact_path("evidence_status.json")
        write_json(path, gate.model_dump(mode="json"))
        self.record_lineage(
            current_missing_evidence=gate.missing_required,
            trusted=gate.trusted,
        )
        return path

    def record_lineage(
        self,
        parent_run_id: str | None = None,
        inherited_missing_evidence: list[str] | None = None,
        current_missing_evidence: list[str] | None = None,
        trusted: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a lineage snapshot for the current run."""
        context_parent = self.context.metadata.get("parent_run_id")
        parent = parent_run_id or (str(context_parent) if context_parent is not None else None)
        inherited = inherited_missing_evidence
        if inherited is None:
            raw_inherited = self.context.metadata.get("inherited_missing_evidence", [])
            inherited = [str(item) for item in raw_inherited] if isinstance(raw_inherited, list) else []
        current = current_missing_evidence
        if current is None:
            current = missing_evidence_from_status(self.context.artifact_path("evidence_status.json"))
        evidence = self.evidence_store.load_run(self.context.run_id)
        merged_metadata = dict(self.context.metadata)
        if metadata:
            merged_metadata.update(metadata)
        self.lineage_store.append(
            build_lineage_record(
                run_id=self.context.run_id,
                run_dir=self.context.run_dir,
                parent_run_id=parent,
                dataset_version=self.context.dataset_version,
                dataset_manifest_sha256=self.context.dataset_manifest_sha256,
                inherited_missing_evidence=inherited,
                current_missing_evidence=current,
                trusted=bool(trusted) if trusted is not None else trusted_from_status(self.context.artifact_path("evidence_status.json")),
                metrics=evidence.metrics,
                metric_records=evidence.metric_records,
                metadata=merged_metadata,
            )
        )

    def next_round_payload(self, raw_plan: dict[str, Any]) -> dict[str, Any]:
        """Build the next-round checklist payload from current evidence state."""
        gate = self.current_gate()
        evidence = self.evidence_store.load_run(self.context.run_id)
        error_facts = ErrorFactStore(self.context.run_root).read(self.context.run_id)
        loop_diagnosis = read_optional_mapping(self.context.artifact_path("loop_diagnosis.json"))
        inherited_missing = context_list(self.context.metadata.get("inherited_missing_evidence", []))
        current_missing = list(gate.missing_required)
        newly_available = [item for item in inherited_missing if item not in set(current_missing)]
        unresolved_diagnoses = unresolved_diagnoses_from_evidence(loop_diagnosis, gate, evidence)
        inherited_unresolved = context_mapping_list(self.context.metadata.get("inherited_unresolved_diagnoses", []))
        diagnosis_delta = diagnosis_delta_from_parent(inherited_unresolved, unresolved_diagnoses)
        return {
            "parent_run_id": self.context.run_id,
            "parent_best_candidate": parent_best_candidate(evidence),
            "dataset_version": self.context.dataset_version,
            "next_dataset_version": self.context.metadata.get("active_learning_next_dataset_version"),
            "unresolved_diagnoses": unresolved_diagnoses,
            "error_facts": error_fact_summary(error_facts),
            "error_fact_action_candidates": error_fact_action_candidates(error_facts),
            "improved_errors": diagnosis_delta["improved_errors"],
            "unresolved_errors": diagnosis_delta["unresolved_errors"],
            "regressed_errors": diagnosis_delta["regressed_errors"],
            "newly_available_evidence": newly_available,
            "recommended_stage": recommended_stage(current_missing, unresolved_diagnoses),
            "stop_reason": next_round_stop_reason(current_missing, unresolved_diagnoses),
            "evidence_delta": {
                "inherited_missing": inherited_missing,
                "current_missing": current_missing,
                "resolved_since_parent": newly_available,
                "present_now": present_evidence_names(gate, evidence),
            },
            "changed_variables": raw_plan.get("changed_variables", {}),
            "evidence_required": raw_plan.get("evidence_required", []),
            "guardrails": raw_plan.get("guardrails", []),
            "status": "ready_for_evidence_collection",
        }


def error_fact_summary(facts: list[ErrorFact], limit: int = 20) -> list[dict[str, Any]]:
    """Return high-signal facts for next-round planning."""
    ranked = sorted(facts, key=_error_fact_rank)
    return [
        {
            "fact_type": fact.fact_type,
            "subject": fact.subject,
            "class_name": fact.class_name,
            "class_pair": fact.class_pair,
            "area": fact.area,
            "metric_name": fact.metric_name,
            "value": fact.value,
            "count": fact.count,
            "rank": fact.rank,
            "severity": fact.severity,
            "action_candidates": fact.action_candidates,
            "candidate_id": fact.candidate_id,
            "node_id": fact.node_id,
        }
        for fact in ranked[:limit]
    ]


def error_fact_action_candidates(facts: list[ErrorFact]) -> list[str]:
    """Return deduplicated actions from medium/high severity facts."""
    actions: list[str] = []
    for fact in sorted(facts, key=_error_fact_rank):
        if fact.severity in {"high", "medium"}:
            actions.extend(fact.action_candidates)
    return list(dict.fromkeys(actions))


def _error_fact_rank(fact: ErrorFact) -> tuple[int, int, float]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}[fact.severity]
    rank = fact.rank if fact.rank is not None else 999
    value = numeric_metric(fact.value)
    score = value if value is not None else 999.0
    return (severity_rank, rank, score)


def loop_plan_evidence_required(path: Path) -> list[str]:
    """Return extra evidence names requested by loop_plan.yaml."""
    if not path.is_file():
        return []
    raw = read_yaml(path)
    values = raw.get("evidence_required", [])
    return [str(value) for value in values] if isinstance(values, list) else []


def missing_evidence_from_status(path: Path) -> list[str]:
    """Return missing evidence names from a persisted evidence gate result."""
    if not path.is_file():
        return []
    raw = read_json(path)
    values = raw.get("missing_required", []) if isinstance(raw, dict) else []
    return [str(value) for value in values] if isinstance(values, list) else []


def trusted_from_status(path: Path) -> bool:
    """Return trusted flag from a persisted evidence gate result."""
    if not path.is_file():
        return False
    raw = read_json(path)
    return bool(raw.get("trusted")) if isinstance(raw, dict) else False


def read_optional_mapping(path: Path) -> dict[str, Any]:
    """Read an optional JSON/YAML mapping artifact."""
    if not path.is_file():
        return {}
    data = read_yaml(path) if path.suffix.lower() in {".yaml", ".yml"} else read_json(path)
    return data if isinstance(data, dict) else {}


def context_list(value: Any) -> list[str]:
    """Coerce a context metadata value to a string list."""
    return [str(item) for item in value] if isinstance(value, list) else []


def context_mapping_list(value: Any) -> list[dict[str, Any]]:
    """Coerce a metadata value to a list of mappings."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def diagnosis_delta_from_parent(
    inherited_unresolved: list[dict[str, Any]],
    current_unresolved: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Compare inherited and current unresolved diagnoses."""
    inherited_by_key = {_diagnosis_key(item): item for item in inherited_unresolved}
    current_by_key = {_diagnosis_key(item): item for item in current_unresolved}
    if not inherited_by_key:
        return {
            "improved_errors": [],
            "unresolved_errors": list(current_unresolved),
            "regressed_errors": [],
        }
    inherited_keys = set(inherited_by_key)
    current_keys = set(current_by_key)
    return {
        "improved_errors": [inherited_by_key[key] for key in sorted(inherited_keys - current_keys)],
        "unresolved_errors": [current_by_key[key] for key in sorted(inherited_keys & current_keys)],
        "regressed_errors": [current_by_key[key] for key in sorted(current_keys - inherited_keys)],
    }


def unresolved_diagnoses_from_evidence(
    loop_diagnosis: dict[str, Any],
    gate: EvidenceGateResult,
    evidence: Evidence,
) -> list[dict[str, Any]]:
    """Return diagnoses that still lack expected evidence."""
    diagnostics = loop_diagnosis.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        return []
    unresolved: list[dict[str, Any]] = []
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        expected = [str(metric) for metric in item.get("expected_metrics", []) if metric is not None]
        missing_expected = [
            metric for metric in expected if not evidence_has(metric, gate, evidence)
        ]
        if missing_expected or not gate.trusted:
            unresolved.append(
                {
                    "category": item.get("category", "unknown"),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "missing_expected_evidence": missing_expected,
                    "next_actions": list(item.get("next_actions", []))
                    if isinstance(item.get("next_actions"), list)
                    else [],
                    "risks": list(item.get("risks", [])) if isinstance(item.get("risks"), list) else [],
                }
            )
    return unresolved


def _diagnosis_key(item: dict[str, Any]) -> str:
    """Return a stable key for a diagnosis/error thread."""
    return "|".join(
        [
            str(item.get("category", "unknown")),
            str(item.get("question", "")),
            str(item.get("answer", "")),
        ]
    )


def present_evidence_names(gate: EvidenceGateResult, evidence: Evidence) -> list[str]:
    """Return evidence names currently present in the run."""
    names: list[str] = []
    names.extend(status.name for status in gate.statuses if status.present)
    names.extend(key for key, value in evidence.metrics.items() if value is not None)
    names.extend(record.metric_name for record in evidence.metric_records if record.value is not None and record.verified)
    names.extend(entry.name for entry in evidence.artifact_manifest if entry.verify())
    return list(dict.fromkeys(str(name) for name in names))


def evidence_has(name: str, gate: EvidenceGateResult, evidence: Evidence) -> bool:
    """Return whether a metric/artifact evidence name is present."""
    if any(status.name == name and status.present for status in gate.statuses):
        return True
    if evidence.metrics.get(name) is not None:
        return True
    if any(record.metric_name == name and record.value is not None and record.verified for record in evidence.metric_records):
        return True
    return any(entry.name == name and entry.verify() for entry in evidence.artifact_manifest)


def parent_best_candidate(evidence: Evidence) -> dict[str, Any] | None:
    """Return the best evidence-backed candidate for the parent run."""
    metric_record = best_metric_record(evidence.metric_records)
    if metric_record is not None:
        return {
            "candidate_id": metric_record.candidate_id,
            "node_id": metric_record.node_id,
            "metric_name": metric_record.metric_name,
            "metric_value": metric_record.value,
            "source": metric_record.source,
        }
    metric_name, metric_value = best_run_metric(evidence.metrics)
    if metric_name is None:
        return None
    return {
        "candidate_id": evidence.run_id,
        "node_id": None,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "source": "run_metrics",
    }


def best_metric_record(records: list[MetricEvidence]) -> MetricEvidence | None:
    """Return the highest-priority trusted candidate metric."""
    preferred = ["map50", "mAP", "map", "map50_95", "recall"]
    index = EvidenceIndex(records)
    for metric_name in preferred:
        record = index.select_best(metric_name=metric_name, verified=True)
        if record is not None and numeric_metric(record.value) is not None:
            return record
    return None


def best_run_metric(metrics: dict[str, MetricValue]) -> tuple[str | None, float | None]:
    """Return the best run-level metric when node metrics are unavailable."""
    for metric_name in ["map50", "mAP", "map", "map50_95", "recall"]:
        value = numeric_metric(metrics.get(metric_name))
        if value is not None:
            return metric_name, value
    return None, None


def numeric_metric(value: MetricValue) -> float | None:
    """Coerce numeric metric values while excluding bools."""
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def recommended_stage(current_missing: list[str], unresolved_diagnoses: list[dict[str, Any]]) -> str:
    """Recommend where the child run should focus first."""
    missing = set(current_missing)
    if "dataset_report" in missing:
        return "profile_data"
    if "label_quality_report" in missing:
        return "advise_labels"
    if "smoke_result" in missing:
        return "smoke"
    if missing.intersection({"latency_ms", "map50", "recall", "precision", "model_size_mb"}):
        return "import_metrics"
    if unresolved_diagnoses:
        return "generate_loop_plan"
    return "report"


def next_round_stop_reason(current_missing: list[str], unresolved_diagnoses: list[dict[str, Any]]) -> str:
    """Explain why another run should exist."""
    if current_missing:
        return "missing_evidence"
    if unresolved_diagnoses:
        return "unresolved_diagnoses"
    return "evidence_complete"
