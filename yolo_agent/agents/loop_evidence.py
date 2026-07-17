"""Evidence gates, lineage snapshots, and next-round evidence deltas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from yolo_agent.agents.loop_io import read_json, read_yaml, write_json
from yolo_agent.agents.diagnosis_graph import DiagnosisGraph
from yolo_agent.agents.doctor_report import build_doctor_decision_report, merge_evidence_grounded_doctor_report
from yolo_agent.agents.policy_learner import PolicyLearner
from yolo_agent.core.coco_error_selection import select_coco_error_facts
from yolo_agent.core.evidence_contract import EvidenceGate, EvidenceGateResult, default_loop_evidence_requirements
from yolo_agent.core.evidence_index import EvidenceIndex
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.error_facts import ErrorFact, ErrorFactStore
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue
from yolo_agent.core.loop_state import LoopState
from yolo_agent.core.policy_memory import PolicyMemoryStore
from yolo_agent.core.paired_experiment import build_paired_experiment_result
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
        error_fact_store = ErrorFactStore(self.context.run_root)
        all_error_facts = error_fact_store.read(self.context.run_id)
        matched_controls = [fact for fact in all_error_facts if fact.evidence_role == "baseline_reference"]
        error_facts = [fact for fact in all_error_facts if fact.evidence_role == "current_observation"]
        parent_error_facts = matched_controls or _parent_error_facts(self.context, error_fact_store)
        diagnosis_graph = DiagnosisGraph.from_yaml().diagnose(error_facts)
        error_delta = error_fact_delta(parent_error_facts, error_facts)
        changed_variables = raw_plan.get("changed_variables", self.context.metadata.get("inherited_changed_variables", {}))
        if not isinstance(changed_variables, dict):
            changed_variables = {}
        parent_evidence = _parent_evidence(self.context, self.evidence_store)
        policy_memory_store = PolicyMemoryStore(self.context.run_root)
        action_context = policy_memory_action_context(self.context, raw_plan, evidence)
        current_primary = [
            record
            for record in evidence.metric_records
            if record.run_id == self.context.run_id
            and (record.origin_run_id or record.run_id) == self.context.run_id
            and record.evidence_role == "current_observation"
            and record.inheritance_depth == 0
            and record.metric_name in {"map50_95", "coco_ap50_95"}
            and record.verified
        ]
        paired_result = None
        if current_primary:
            current_record = max(current_primary, key=lambda item: item.created_at)
            paired_result = build_paired_experiment_result(
                run_id=self.context.run_id,
                candidate_id=current_record.candidate_id,
                candidate_node_id=current_record.node_id,
                metric_records=evidence.metric_records,
                error_facts=all_error_facts,
                primary_metric="map50_95",
                target_error_facts=[fact.model_dump(mode="json") for fact in error_facts],
            )
        learned_policy_records = PolicyLearner(policy_memory_store).learn_from_error_delta(
            run_id=self.context.run_id,
            parent_run_id=_parent_run_id(self.context),
            dataset_version=self.context.dataset_version,
            error_delta=error_delta,
            current_evidence=evidence,
            parent_evidence=parent_evidence,
            changed_variables=changed_variables,
            scenario=_task_scene(self.context.task_path),
            recipe_id=action_context["recipe_id"],
            recipe_version=action_context["recipe_version"],
            component_versions=action_context["component_versions"],
            model_family=action_context["model_family"],
            dataset_signature=action_context["dataset_signature"],
            protocol_hash=action_context["protocol_hash"],
            fidelity=action_context["fidelity"],
            seed=action_context["seed"],
            action_before_values=action_context["action_before_values"],
            paired_result=paired_result,
        )
        policy_memory_summary = policy_memory_store.summarize(dataset_version=self.context.dataset_version)
        coco_selection = select_coco_error_facts(error_facts)
        error_delta_policy = error_delta_next_round_policy(
            parent_error_facts=parent_error_facts,
            current_error_facts=error_facts,
            error_delta=error_delta,
            baseline_focus=[item.model_dump(mode="json") for item in coco_selection.current_round_focus],
            baseline_actions=coco_selection.focus_action_candidates,
        )
        loop_diagnosis = read_optional_mapping(self.context.artifact_path("loop_diagnosis.json"))
        inherited_missing = context_list(self.context.metadata.get("inherited_missing_evidence", []))
        current_missing = list(gate.missing_required)
        newly_available = [item for item in inherited_missing if item not in set(current_missing)]
        unresolved_diagnoses = unresolved_diagnoses_from_evidence(loop_diagnosis, gate, evidence)
        inherited_unresolved = context_mapping_list(self.context.metadata.get("inherited_unresolved_diagnoses", []))
        diagnosis_delta = diagnosis_delta_from_parent(inherited_unresolved, unresolved_diagnoses)
        doctor_report = build_doctor_decision_report(
            diagnosis_graph=diagnosis_graph,
            current_round_focus=error_delta_policy["current_round_focus"],
            current_round_error_actions=dedupe_strings(
                [
                    *error_delta_policy["current_round_error_actions"],
                    *diagnosis_graph.action_candidates,
                ]
            ),
            error_delta_policy=error_delta_policy,
            error_delta=error_delta,
            raw_plan=raw_plan,
            current_missing_evidence=current_missing,
            newly_available_evidence=newly_available,
        )
        doctor_report = merge_evidence_grounded_doctor_report(
            rule_report=doctor_report,
            llm_draft=llm_doctor_report_draft(self.context.artifact_path("llm_decision.yaml")),
            evidence=evidence,
            error_facts=error_facts,
        )
        return {
            "parent_run_id": self.context.run_id,
            "parent_best_candidate": parent_best_candidate(evidence),
            "dataset_version": self.context.dataset_version,
            "next_dataset_version": self.context.metadata.get("active_learning_next_dataset_version"),
            "unresolved_diagnoses": unresolved_diagnoses,
            "error_facts": error_fact_summary(error_facts),
            "diagnosis_graph": diagnosis_graph.model_dump(mode="json"),
            "diagnosis_graph_evidence_needed": diagnosis_graph.evidence_needed,
            "diagnosis_graph_action_candidates": diagnosis_graph.action_candidates,
            "error_fact_action_candidates": error_fact_action_candidates(error_facts),
            "coco_error_selection": coco_selection.model_dump(mode="json"),
            "top_unresolved_diagnoses": [
                item.model_dump(mode="json") for item in coco_selection.top_unresolved_diagnoses
            ],
            "current_round_focus": error_delta_policy["current_round_focus"],
            "current_round_error_actions": error_delta_policy["current_round_error_actions"],
            "error_delta_proposal_policy": error_delta_policy,
            "doctor_report": doctor_report.model_dump(mode="json"),
            "proposal_mode": error_delta_policy["proposal_mode"],
            "proposal_budget_profiles_allowed": error_delta_policy["proposal_budget_profiles_allowed"],
            "proposal_budget_profiles_blocked": error_delta_policy["proposal_budget_profiles_blocked"],
            "proposal_required_bindings": error_delta_policy["proposal_required_bindings"],
            "full_candidate_proposal_allowed": error_delta_policy["full_candidate_proposal_allowed"],
            "pilot_candidate_proposal_allowed": error_delta_policy["pilot_candidate_proposal_allowed"],
            "error_fact_delta": error_delta,
            "improved_error_facts": error_delta["improved_errors"],
            "unresolved_error_facts": error_delta["unresolved_errors"],
            "regressed_error_facts": error_delta["regressed_errors"],
            "next_error_actions": error_delta["next_action_candidates"],
            "effective_error_actions": error_delta["effective_action_candidates"],
            "policy_memory_path": policy_memory_store.path.as_posix(),
            "policy_memory_records_created": len(learned_policy_records),
            "policy_memory_summary": [
                item.model_dump(mode="json") for item in policy_memory_summary[:20]
            ],
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
            "changed_variables": changed_variables,
            "evidence_required": raw_plan.get("evidence_required", []),
            "guardrails": dedupe_strings([*raw_plan.get("guardrails", []), *error_delta_policy["guardrails"]]),
            "status": error_delta_policy["status"],
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


def error_fact_delta(parent_facts: list[ErrorFact], current_facts: list[ErrorFact]) -> dict[str, Any]:
    """Compare error facts only when they share an exact matched-control identity."""
    eligible_parents = [fact for fact in parent_facts if fact.evidence_role == "baseline_reference"]
    eligible_current = [fact for fact in current_facts if fact.evidence_role == "current_observation"]
    parent_by_key = {(_error_fact_key(fact), _error_fact_match_hash(fact)): fact for fact in eligible_parents if _error_fact_match_hash(fact)}
    current_by_key = {(_error_fact_key(fact), _error_fact_match_hash(fact)): fact for fact in eligible_current if _error_fact_match_hash(fact)}
    parent_keys = set(parent_by_key)
    current_keys = set(current_by_key)
    improved: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for key in sorted(parent_keys | current_keys):
        parent = parent_by_key.get(key)
        current = current_by_key.get(key)
        if parent is not None and current is None:
            improved.append(_error_fact_delta_item(parent, None, "resolved"))
            continue
        if parent is None and current is not None:
            item = _error_fact_delta_item(None, current, "new")
            regressed.append(item)
            if current.severity in {"high", "medium"}:
                unresolved.append(item)
            continue
        if parent is None or current is None:
            continue
        trend = _error_fact_trend(parent, current)
        item = _error_fact_delta_item(parent, current, trend)
        if trend == "improved":
            improved.append(item)
        elif trend == "regressed":
            regressed.append(item)
        else:
            unchanged.append(item)
        if current.severity in {"high", "medium"} and trend != "improved":
            unresolved.append(item)

    if not eligible_parents:
        unresolved = [
            _error_fact_delta_item(None, fact, "current")
            for fact in sorted(eligible_current, key=_error_fact_rank)
            if fact.severity in {"high", "medium"}
        ]

    return {
        "parent_fact_count": len(eligible_parents),
        "current_fact_count": len(eligible_current),
        "comparison_mode": "paired_matched_baseline",
        "needs_matched_baseline": bool(current_facts) and not bool(parent_by_key),
        "improved_errors": improved,
        "unchanged_errors": unchanged,
        "regressed_errors": regressed,
        "unresolved_errors": sorted(unresolved, key=_delta_item_rank),
        "effective_action_candidates": _actions_from_delta(improved),
        "next_action_candidates": _actions_from_delta(unresolved),
    }


def _error_fact_match_hash(fact: ErrorFact) -> str | None:
    fields = {
        "dataset_manifest_sha256": fact.dataset_manifest_sha256,
        "subset_manifest_sha256": fact.subset_manifest_sha256,
        "seed": None if fact.seed is None else str(fact.seed),
        "epochs": fact.epochs,
        "fidelity": fact.fidelity,
        "batch_policy_hash": fact.batch_policy_hash,
        "ultralytics_version": fact.ultralytics_version,
        "imgsz": fact.imgsz,
        "eval_protocol_hash": fact.eval_protocol_hash,
        "split": fact.split,
    }
    if any(value is None or value == "" for value in fields.values()):
        return None
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def error_delta_next_round_policy(
    parent_error_facts: list[ErrorFact],
    current_error_facts: list[ErrorFact],
    error_delta: dict[str, Any],
    baseline_focus: list[dict[str, Any]],
    baseline_actions: list[str],
) -> dict[str, Any]:
    """Return the executable next-round policy derived from error facts/delta.

    First COCO round can only produce pilot proposals from baseline error facts.
    Forked/child rounds must focus on unresolved or regressed parent-current
    error deltas. Full candidate proposals remain guarded by later baseline and
    candidate-promotion gates; they are never allowed without error facts.
    """
    if not current_error_facts:
        return {
            "proposal_mode": "blocked",
            "status": "blocked_missing_error_facts",
            "focus_source": "none",
            "current_round_focus": [],
            "current_round_error_actions": [],
            "pilot_candidate_proposal_allowed": False,
            "full_candidate_proposal_allowed": False,
            "proposal_budget_profiles_allowed": [],
            "proposal_budget_profiles_blocked": ["candidate_full"],
            "proposal_required_bindings": ["target_error_facts", "expected_improvement"],
            "rejection_reasons": ["missing_error_facts"],
            "guardrails": [
                "missing_error_facts",
                "import_coco_error_facts_before_generating_candidate_proposals",
                "no_full_candidate_without_error_facts",
            ],
        }

    if parent_error_facts:
        delta_focus = _focus_from_error_delta(error_delta)
        delta_actions = _actions_from_delta(delta_focus)
        if not delta_focus:
            return {
                "proposal_mode": "blocked",
                "status": "blocked_no_unresolved_error_delta",
                "focus_source": "parent_current_error_delta",
                "current_round_focus": [],
                "current_round_error_actions": [],
                "pilot_candidate_proposal_allowed": False,
                "full_candidate_proposal_allowed": False,
                "proposal_budget_profiles_allowed": [],
                "proposal_budget_profiles_blocked": ["candidate_full"],
                "proposal_required_bindings": ["target_error_facts", "expected_improvement"],
                "rejection_reasons": ["no_unresolved_or_regressed_error_delta"],
                "guardrails": [
                    "no_generic_next_round_without_unresolved_error_delta",
                    "do_not_generate_full_candidate_when_errors_are_resolved",
                ],
            }
        return {
            "proposal_mode": "pilot_only",
            "status": "ready_for_error_delta_pilot_proposals",
            "focus_source": "parent_current_error_delta",
            "current_round_focus": delta_focus,
            "current_round_error_actions": delta_actions,
            "pilot_candidate_proposal_allowed": True,
            "full_candidate_proposal_allowed": False,
            "proposal_budget_profiles_allowed": ["debug", "pilot"],
            "proposal_budget_profiles_blocked": ["candidate_full"],
            "proposal_required_bindings": ["target_error_facts", "expected_improvement"],
            "rejection_reasons": [],
            "guardrails": [
                "base_next_round_on_unresolved_or_regressed_error_delta",
                "pilot_only_until_candidate_promotion_gate_passes",
                "no_full_candidate_without_error_facts",
            ],
        }

    if not baseline_focus:
        return {
            "proposal_mode": "blocked",
            "status": "blocked_missing_baseline_error_focus",
            "focus_source": "baseline_error_facts",
            "current_round_focus": [],
            "current_round_error_actions": [],
            "pilot_candidate_proposal_allowed": False,
            "full_candidate_proposal_allowed": False,
            "proposal_budget_profiles_allowed": [],
            "proposal_budget_profiles_blocked": ["candidate_full"],
            "proposal_required_bindings": ["target_error_facts", "expected_improvement"],
            "rejection_reasons": ["missing_baseline_error_focus"],
            "guardrails": [
                "baseline_error_facts_exist_but_no_medium_or_high_focus_was_selected",
                "no_full_candidate_without_target_error_focus",
            ],
        }

    return {
        "proposal_mode": "pilot_only",
        "status": "ready_for_baseline_error_pilot_proposals",
        "focus_source": "baseline_error_facts",
        "current_round_focus": baseline_focus,
        "current_round_error_actions": list(baseline_actions),
        "pilot_candidate_proposal_allowed": True,
        "full_candidate_proposal_allowed": False,
        "proposal_budget_profiles_allowed": ["debug", "pilot"],
        "proposal_budget_profiles_blocked": ["candidate_full"],
        "proposal_required_bindings": ["target_error_facts", "expected_improvement"],
        "rejection_reasons": [],
        "guardrails": [
            "first_error_fact_round_is_pilot_only",
            "candidate_full_requires_later_error_delta_and_candidate_promotion",
            "no_full_candidate_without_error_facts",
        ],
    }


def _error_fact_rank(fact: ErrorFact) -> tuple[int, int, float]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}[fact.severity]
    rank = fact.rank if fact.rank is not None else 999
    value = numeric_metric(fact.value)
    score = value if value is not None else 999.0
    return (severity_rank, rank, score)


def _parent_error_facts(context: RunContext, store: ErrorFactStore) -> list[ErrorFact]:
    """Load parent error facts when the current run is forked from a parent."""
    parent_run_id = context.metadata.get("parent_run_id")
    if not parent_run_id:
        return []
    try:
        return store.read(str(parent_run_id))
    except ValueError:
        return []


def _parent_run_id(context: RunContext) -> str | None:
    """Return parent run id from context metadata."""
    parent_run_id = context.metadata.get("parent_run_id")
    return str(parent_run_id) if parent_run_id else None


def _parent_evidence(context: RunContext, store: EvidenceStore) -> Evidence | None:
    """Load parent evidence when available."""
    parent_run_id = _parent_run_id(context)
    if parent_run_id is None:
        return None
    try:
        return store.load_run(parent_run_id)
    except FileNotFoundError:
        return None


def _task_scene(task_path: Path) -> str | None:
    """Read the task scene without forcing TaskSpec validation here."""
    if not task_path.is_file():
        return None
    try:
        data = read_yaml(task_path)
    except (OSError, ValueError):
        return None
    scene = data.get("scene")
    return str(scene) if scene else None


def _error_fact_key(fact: ErrorFact) -> tuple[str, str, str, str, str, str]:
    """Return a stable identity for comparing an error thread across runs."""
    return (
        fact.fact_type,
        fact.subject,
        fact.class_name or "",
        fact.class_pair or "",
        fact.area or "",
        fact.metric_name or "",
    )


def _error_fact_trend(parent: ErrorFact, current: ErrorFact) -> str:
    """Return improved/unchanged/regressed for one matched error fact."""
    parent_value = _fact_compare_value(parent)
    current_value = _fact_compare_value(current)
    if parent_value is None or current_value is None:
        if _severity_score(current.severity) < _severity_score(parent.severity):
            return "improved"
        if _severity_score(current.severity) > _severity_score(parent.severity):
            return "regressed"
        return "unchanged"
    delta = current_value - parent_value
    if abs(delta) <= 1e-9:
        return "unchanged"
    if _higher_is_better(parent):
        return "improved" if delta > 0 else "regressed"
    return "improved" if delta < 0 else "regressed"


def _fact_compare_value(fact: ErrorFact) -> float | None:
    value = numeric_metric(fact.value)
    if value is not None:
        return value
    return float(fact.count) if fact.count is not None else None


def _higher_is_better(fact: ErrorFact) -> bool:
    if fact.count is not None and fact.value is None:
        return False
    if fact.fact_type in {
        "false_negative_heavy_class",
        "localization_heavy_class",
        "class_confusion_pair",
        "background_false_positive_class",
    }:
        return False
    return True


def _severity_score(severity: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(severity, 1)


def _error_fact_delta_item(parent: ErrorFact | None, current: ErrorFact | None, trend: str) -> dict[str, Any]:
    fact = current or parent
    if fact is None:
        return {}
    parent_value = _fact_compare_value(parent) if parent is not None else None
    current_value = _fact_compare_value(current) if current is not None else None
    return {
        "trend": trend,
        "fact_type": fact.fact_type,
        "subject": fact.subject,
        "class_name": fact.class_name,
        "class_pair": fact.class_pair,
        "area": fact.area,
        "metric_name": fact.metric_name,
        "parent_value": parent_value,
        "current_value": current_value,
        "delta": (
            round(current_value - parent_value, 6)
            if parent_value is not None and current_value is not None
            else None
        ),
        "parent_severity": parent.severity if parent is not None else None,
        "current_severity": current.severity if current is not None else None,
        "action_candidates": (current or parent).action_candidates,
        "candidate_id": fact.candidate_id,
        "node_id": fact.node_id,
        "matched_control_hash": _error_fact_match_hash(fact),
    }


def _delta_item_rank(item: dict[str, Any]) -> tuple[int, float]:
    severity = str(item.get("current_severity") or item.get("parent_severity") or "medium")
    value = item.get("current_value")
    numeric = float(value) if isinstance(value, (int, float)) else 999.0
    return (-_severity_score(severity), numeric)


def _actions_from_delta(items: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for item in items:
        raw_actions = item.get("action_candidates", [])
        if isinstance(raw_actions, list):
            actions.extend(str(action) for action in raw_actions)
    return list(dict.fromkeys(actions))


def _focus_from_error_delta(error_delta: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unresolved/regressed delta items as next-round focus rows."""
    raw_items = [
        *error_delta.get("regressed_errors", []),
        *error_delta.get("unresolved_errors", []),
    ]
    deduped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        trend = str(item.get("trend", "unresolved"))
        if trend == "improved":
            continue
        key = (
            str(item.get("fact_type", "")),
            str(item.get("subject", "")),
            str(item.get("class_name", "")),
            str(item.get("area", "")),
            str(item.get("metric_name", "")),
        )
        focus = dict(item)
        focus["diagnosis_kind"] = _delta_diagnosis_kind(focus)
        focus["focus_source"] = "parent_current_error_delta"
        focus["reason"] = _delta_focus_reason(focus)
        deduped.setdefault(key, focus)
    return sorted(deduped.values(), key=_delta_item_rank)


def _delta_diagnosis_kind(item: dict[str, Any]) -> str:
    fact_type = str(item.get("fact_type", ""))
    area = str(item.get("area", ""))
    metric_name = str(item.get("metric_name", ""))
    if fact_type in {"area_metric", "subset_performance"} and area == "small":
        return "small_object_ap"
    if fact_type == "per_class_metric" and metric_name == "per_class_ar":
        return "class_recall"
    if fact_type == "class_low_ap":
        return "class_low_ap"
    if fact_type == "false_negative_heavy_class":
        return "false_negative_class"
    if fact_type == "localization_heavy_class":
        return "localization_class"
    if fact_type == "background_false_positive_class":
        return "background_fp_class"
    if fact_type == "class_confusion_pair":
        return "class_confusion"
    return "generic_error_fact"


def _delta_focus_reason(item: dict[str, Any]) -> str:
    subject = str(item.get("class_name") or item.get("class_pair") or item.get("area") or item.get("subject") or "error")
    trend = str(item.get("trend", "unresolved"))
    if trend == "regressed":
        return f"{subject} regressed versus the parent run and should drive the next pilot."
    if trend == "new":
        return f"{subject} is a new error fact in the current run."
    return f"{subject} remains unresolved versus the parent run."


def dedupe_strings(values: list[Any]) -> list[str]:
    """Return strings with stable de-duplication."""
    return list(dict.fromkeys(str(value) for value in values if value is not None))


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


def llm_doctor_report_draft(path: Path) -> dict[str, Any] | None:
    """Return the optional LLM doctor draft artifact."""
    raw = read_optional_mapping(path)
    draft = raw.get("doctor_report_draft")
    if isinstance(draft, dict):
        return draft
    bundle = raw.get("proposal_bundle")
    if isinstance(bundle, dict):
        nested = bundle.get("doctor_report_draft")
        if isinstance(nested, dict):
            return nested
    return None


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


def policy_memory_action_context(
    context: RunContext,
    raw_plan: dict[str, Any],
    evidence: Evidence,
) -> dict[str, Any]:
    """Extract normalized action identity context without trusting candidate names."""
    policies = raw_plan.get("candidate_policies", [])
    policy = policies[0] if isinstance(policies, list) and len(policies) == 1 and isinstance(policies[0], dict) else {}
    components = policy.get("components", []) if isinstance(policy, dict) else []
    component_versions = policy.get("component_versions", {}) if isinstance(policy, dict) else {}
    if not isinstance(component_versions, dict):
        component_versions = {}
    for component_id in components if isinstance(components, list) else []:
        component_versions.setdefault(str(component_id), "unknown")
    model = str(
        context.metadata.get("training_model")
        or (policy.get("base_model") if isinstance(policy, dict) else "")
        or "unknown"
    )
    current_records = [
        record
        for record in evidence.metric_records
        if record.evidence_role == "current_observation" and record.inheritance_depth == 0
    ]
    latest_record = max(current_records, key=lambda item: item.created_at) if current_records else None
    profile = str(context.metadata.get("training_profile") or "unknown")
    raw_fidelity = str(latest_record.fidelity if latest_record is not None else profile)
    fidelity_aliases = {
        "baseline_full": "full",
        "baseline_confirm": "full",
        "candidate_full_seed_1": "candidate_full",
        "candidate_full_confirmation": "candidate_full",
    }
    fidelity = fidelity_aliases.get(raw_fidelity, raw_fidelity)
    if fidelity not in {"debug", "pilot", "pilot_3", "pilot_10", "candidate_full", "full"}:
        fidelity = "unknown"
    protocol_hash = str(context.metadata.get("baseline_protocol_hash") or "")
    if not protocol_hash:
        protocol_hash = next(
            (
                str(record.protocol_hash)
                for record in evidence.metric_records
                if record.protocol_hash and record.evidence_role == "current_observation"
            ),
            "unknown",
        )
    fixed_variables = policy.get("fixed_variables", {}) if isinstance(policy, dict) else {}
    return {
        "recipe_id": str(policy.get("action_id") or policy.get("policy_id") or "") or None,
        "recipe_version": str(policy.get("recipe_version") or policy.get("version") or "unknown"),
        "component_versions": {str(key): str(value) for key, value in component_versions.items()},
        "model_family": _normalize_model_family(model),
        "dataset_signature": context.dataset_manifest_sha256 or context.dataset_version,
        "protocol_hash": protocol_hash,
        "fidelity": fidelity,
        "seed": latest_record.seed if latest_record is not None and latest_record.seed is not None else "unknown",
        "action_before_values": fixed_variables if isinstance(fixed_variables, dict) else {},
    }


def _normalize_model_family(model: str) -> str:
    lowered = model.lower()
    for family in ("yolo26", "yolo11", "yolov10", "yolov9", "yolov8"):
        if family in lowered:
            return family
    return lowered.rsplit("/", 1)[-1].split(".", 1)[0] or "unknown"
