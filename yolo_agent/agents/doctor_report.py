"""Doctor-style decision reports for optimization loop rounds."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.diagnosis_graph import DiagnosisGraphReport
from yolo_agent.core.error_facts import ErrorFact
from yolo_agent.core.experiment_graph import Evidence
from yolo_agent.utils import dedupe_list


ActionType = Literal[
    "run_training",
    "import_metrics",
    "mine_errors",
    "profile_data",
    "advise_labels",
    "benchmark_latency",
    "data_action",
    "label_action",
    "augmentation_action",
    "postprocess_action",
]


class RejectedAction(BaseModel):
    """One action rejected by constraints, missing evidence, or loop policy."""

    action: str
    reason: str


class SelectedAction(BaseModel):
    """One action selected for the next round."""

    action: str
    action_type: ActionType = "run_training"
    target: str | None = None
    why: list[str] = Field(default_factory=list)


class DoctorDecisionReport(BaseModel):
    """Human-readable diagnosis-to-action summary for one loop round."""

    primary_problem: str
    likely_causes: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    rejected_actions: list[RejectedAction] = Field(default_factory=list)
    selected_actions: list[SelectedAction] = Field(default_factory=list)
    why: list[str] = Field(default_factory=list)
    expected_improvement: dict[str, str] = Field(default_factory=dict)
    stop_condition: list[str] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "low"
    status: str = "unknown"
    missing_evidence: list[str] = Field(default_factory=list)
    target_error_facts: list[dict[str, Any]] = Field(default_factory=list)
    llm_merge: dict[str, Any] = Field(default_factory=dict)


def build_doctor_decision_report(
    *,
    diagnosis_graph: DiagnosisGraphReport,
    current_round_focus: list[dict[str, Any]],
    current_round_error_actions: list[str],
    error_delta_policy: dict[str, Any],
    error_delta: dict[str, Any],
    raw_plan: dict[str, Any],
    current_missing_evidence: list[str],
    newly_available_evidence: list[str],
) -> DoctorDecisionReport:
    """Build a concise doctor-style decision summary from loop facts."""
    primary_focus = current_round_focus[0] if current_round_focus else {}
    finding = diagnosis_graph.findings[0] if diagnosis_graph.findings else None
    primary_problem = _primary_problem(primary_focus, finding.symptom if finding else None)
    likely_causes = _likely_causes(diagnosis_graph)
    evidence = _evidence_lines(current_round_focus, diagnosis_graph, newly_available_evidence)
    rejected = _rejected_actions(raw_plan, error_delta_policy)
    selected = _selected_actions(current_round_error_actions, current_round_focus, current_missing_evidence)
    why = _why_lines(
        primary_focus=primary_focus,
        finding_rationale=finding.rationale if finding else "",
        selected=selected,
        error_delta_policy=error_delta_policy,
    )
    expected = _expected_improvement(current_round_focus, selected)
    return DoctorDecisionReport(
        primary_problem=primary_problem,
        likely_causes=likely_causes,
        evidence=evidence,
        rejected_actions=rejected,
        selected_actions=selected,
        why=why,
        expected_improvement=expected,
        stop_condition=_stop_conditions(current_missing_evidence, current_round_focus, selected, error_delta_policy),
        confidence=_confidence(current_round_focus, current_missing_evidence, diagnosis_graph, error_delta),
        status=str(error_delta_policy.get("status", "unknown")),
        missing_evidence=list(current_missing_evidence),
        target_error_facts=current_round_focus,
    )


def merge_evidence_grounded_doctor_report(
    *,
    rule_report: DoctorDecisionReport,
    llm_draft: dict[str, Any] | None,
    evidence: Evidence,
    error_facts: list[ErrorFact],
) -> DoctorDecisionReport:
    """Merge an LLM doctor draft into the rule report only when evidence-grounded.

    The rule report remains authoritative. LLM evidence lines are accepted only
    when they can be traced to error facts, verified metric evidence, run metrics,
    or verified artifact manifest entries.
    """
    if not isinstance(llm_draft, dict) or not llm_draft:
        return rule_report.model_copy(update={"llm_merge": {"used": False, "reason": "missing_llm_doctor_report_draft"}})

    catalog = _grounding_catalog(rule_report, evidence, error_facts)
    accepted_evidence, rejected_evidence = _grounded_lines(llm_draft.get("evidence", []), catalog)
    grounded_context = _catalog_tokens(catalog)
    grounded_context.update(_tokens(" ".join([*rule_report.evidence, *accepted_evidence])))

    accepted_why, rejected_why = _grounded_explanations(
        llm_draft.get("why", []),
        grounded_context,
    )
    accepted_causes, rejected_causes = _grounded_explanations(
        llm_draft.get("likely_causes", []),
        grounded_context,
    )
    accepted_stop, rejected_stop = _grounded_explanations(
        llm_draft.get("stop_condition", []),
        grounded_context,
    )
    selected_actions, rejected_selected_actions = _grounded_selected_actions(
        llm_draft.get("selected_actions", []),
        rule_report,
        error_facts,
    )
    rejected_actions, rejected_rejected_actions = _grounded_rejected_actions(
        llm_draft.get("rejected_actions", []),
        rule_report,
    )

    llm_expected, rejected_expected = _grounded_expected_improvement(
        llm_draft.get("expected_improvement", {}),
        grounded_context,
    )
    expected_improvement = dict(rule_report.expected_improvement)
    for metric, value in llm_expected.items():
        expected_improvement.setdefault(metric, value)

    return rule_report.model_copy(
        update={
            "evidence": dedupe_list([*rule_report.evidence, *accepted_evidence])[:12],
            "likely_causes": dedupe_list([*rule_report.likely_causes, *accepted_causes])[:8],
            "why": dedupe_list([*rule_report.why, *accepted_why])[:10],
            "stop_condition": dedupe_list([*rule_report.stop_condition, *accepted_stop])[:8],
            "selected_actions": _dedupe_selected([*rule_report.selected_actions, *selected_actions])[:10],
            "rejected_actions": _dedupe_rejected([*rule_report.rejected_actions, *rejected_actions])[:10],
            "expected_improvement": expected_improvement,
            "llm_merge": {
                "used": True,
                "accepted_evidence": accepted_evidence,
                "rejected_evidence": rejected_evidence,
                "accepted_why": accepted_why,
                "rejected_why": rejected_why,
                "accepted_likely_causes": accepted_causes,
                "rejected_likely_causes": rejected_causes,
                "accepted_stop_condition": accepted_stop,
                "rejected_stop_condition": rejected_stop,
                "accepted_selected_actions": [item.action for item in selected_actions],
                "rejected_selected_actions": rejected_selected_actions,
                "accepted_rejected_actions": [item.action for item in rejected_actions],
                "rejected_rejected_actions": rejected_rejected_actions,
                "accepted_expected_improvement": llm_expected,
                "rejected_expected_improvement": rejected_expected,
            },
        }
    )


def _primary_problem(primary_focus: dict[str, Any], fallback_symptom: str | None) -> str:
    kind = str(primary_focus.get("diagnosis_kind", ""))
    subject = str(
        primary_focus.get("class_name")
        or primary_focus.get("class_pair")
        or primary_focus.get("area")
        or primary_focus.get("subject")
        or ""
    )
    if kind == "small_object_ap":
        return "AP_small low"
    if kind == "medium_object_ap":
        return "AP_medium low"
    if kind == "large_object_ap":
        return "AP_large low"
    if kind == "class_recall":
        return f"{subject} recall low" if subject else "Class recall low"
    if kind == "class_low_ap":
        return f"{subject} AP low" if subject else "Per-class AP low"
    if kind == "false_negative_class":
        return f"{subject} false negatives high" if subject else "False negatives high"
    if kind == "localization_class":
        return f"{subject} localization error high" if subject else "Localization error high"
    if kind == "background_fp_class":
        return f"{subject} background false positives high" if subject else "Background false positives high"
    if kind == "class_confusion":
        return f"{subject} class confusion" if subject else "Class confusion"
    if fallback_symptom:
        return fallback_symptom
    return "Insufficient evidence for primary problem"


def _likely_causes(report: DiagnosisGraphReport) -> list[str]:
    causes: list[str] = []
    for finding in report.findings[:2]:
        for cause in finding.possible_causes:
            causes.append(cause.description)
    return dedupe_list(causes)[:6]


def _evidence_lines(
    focus_items: list[dict[str, Any]],
    diagnosis_graph: DiagnosisGraphReport,
    newly_available_evidence: list[str],
) -> list[str]:
    lines: list[str] = []
    for item in focus_items[:6]:
        line = _focus_evidence_line(item)
        if line:
            lines.append(line)
    for finding in diagnosis_graph.findings[:2]:
        for fact in finding.matched_facts[:4]:
            value = _metric_value(fact.value)
            if value is None and fact.count is None:
                continue
            subject = fact.class_name or fact.class_pair or fact.area or fact.subject
            metric = _metric_label(fact.metric_name or fact.fact_type)
            lines.append(f"{metric}={value if value is not None else fact.count} for {subject}")
    for name in newly_available_evidence:
        lines.append(f"{name} is newly available")
    return dedupe_list(lines)[:10]


def _focus_evidence_line(item: dict[str, Any]) -> str:
    metric = str(item.get("metric_name") or item.get("fact_type") or "error_fact")
    value = _metric_value(item.get("current_value", item.get("value")))
    count = item.get("count")
    subject = str(item.get("class_name") or item.get("class_pair") or item.get("area") or item.get("subject") or "target")
    if value is not None:
        return f"{_metric_label(metric)}={value} for {subject}"
    if count is not None:
        return f"{_metric_label(metric)} count={count} for {subject}"
    severity = item.get("current_severity", item.get("severity"))
    return f"{subject} severity={severity}" if severity else ""


def _rejected_actions(raw_plan: dict[str, Any], error_delta_policy: dict[str, Any]) -> list[RejectedAction]:
    rejected: list[RejectedAction] = []
    guardrails = [str(item) for item in raw_plan.get("guardrails", []) if item is not None]
    guardrails.extend(str(item) for item in error_delta_policy.get("guardrails", []) if item is not None)
    for guardrail in guardrails:
        lowered = guardrail.lower()
        if "blocked_imgsz_increase" in lowered or "do_not_increase_imgsz" in lowered:
            rejected.append(
                RejectedAction(
                    action="increase_imgsz",
                    reason=guardrail,
                )
            )
        if "candidate_full" in lowered and "blocked" in lowered:
            rejected.append(
                RejectedAction(
                    action="candidate_full",
                    reason=guardrail,
                )
            )
    for profile in error_delta_policy.get("proposal_budget_profiles_blocked", []):
        rejected.append(
            RejectedAction(
                action=str(profile),
                reason="Blocked by current proposal mode; full runs require trusted evidence and promotion gates.",
            )
        )
    for reason in error_delta_policy.get("rejection_reasons", []):
        rejected.append(
            RejectedAction(
                action="candidate_proposal",
                reason=str(reason),
            )
        )
    return _dedupe_rejected(rejected)


def _selected_actions(
    action_ids: list[str],
    focus_items: list[dict[str, Any]],
    missing_evidence: list[str],
) -> list[SelectedAction]:
    if action_ids:
        return [
            SelectedAction(
                action=action,
                action_type=_action_type(action),
                target=_target_for_action(action, focus_items),
                why=_why_for_action(action, focus_items),
            )
            for action in dedupe_list(action_ids)[:8]
        ]
    evidence_actions = _evidence_actions(missing_evidence)
    return [
        SelectedAction(
            action=action,
            action_type=action,  # type: ignore[arg-type]
            target=", ".join(names),
            why=["Missing evidence blocks a trustworthy training decision."],
        )
        for action, names in evidence_actions.items()
    ]


def _why_lines(
    *,
    primary_focus: dict[str, Any],
    finding_rationale: str,
    selected: list[SelectedAction],
    error_delta_policy: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    reason = primary_focus.get("reason")
    if reason:
        lines.append(str(reason))
    if finding_rationale:
        lines.append(finding_rationale)
    if selected:
        targets = dedupe_list([item.target for item in selected if item.target])
        if targets:
            lines.append(f"Selected actions target {', '.join(targets)} directly.")
    if error_delta_policy.get("proposal_mode") == "pilot_only":
        lines.append("Pilot-only mode limits the next round to cheap, evidence-bound tests.")
    if error_delta_policy.get("full_candidate_proposal_allowed") is False:
        lines.append("Full COCO candidates remain blocked until evidence and promotion gates pass.")
    return dedupe_list(lines)


def _expected_improvement(
    focus_items: list[dict[str, Any]],
    selected: list[SelectedAction],
) -> dict[str, str]:
    if not focus_items or not selected:
        return {"unknown": "Collect missing evidence before estimating improvement."}
    expected: dict[str, str] = {}
    for item in focus_items[:4]:
        metric = str(item.get("metric_name") or item.get("fact_type") or "target_error")
        direction = "decrease" if item.get("fact_type") in {
            "false_negative_heavy_class",
            "localization_heavy_class",
            "class_confusion_pair",
            "background_false_positive_class",
        } else "increase"
        expected[_metric_label(metric)] = f"{direction}; pilot_positive_delta required"
    return expected


def _stop_conditions(
    missing_evidence: list[str],
    focus_items: list[dict[str, Any]],
    selected: list[SelectedAction],
    error_delta_policy: dict[str, Any],
) -> list[str]:
    conditions: list[str] = []
    if missing_evidence:
        conditions.append(f"Missing evidence remains unresolved: {', '.join(missing_evidence)}.")
    if focus_items and selected:
        conditions.append("Pilot does not improve the bound target error facts.")
        conditions.append("Latency or runtime regressions exceed the configured budget.")
    if error_delta_policy.get("proposal_mode") == "blocked":
        conditions.append("No target error facts are available for a trustworthy candidate proposal.")
    if not conditions:
        conditions.append("Evidence is complete and no unresolved diagnosis remains.")
    return dedupe_list(conditions)


def _confidence(
    focus_items: list[dict[str, Any]],
    missing_evidence: list[str],
    diagnosis_graph: DiagnosisGraphReport,
    error_delta: dict[str, Any],
) -> Literal["low", "medium", "high"]:
    if missing_evidence:
        return "low"
    if focus_items and diagnosis_graph.findings and error_delta.get("parent_fact_count", 0):
        return "high"
    if focus_items or diagnosis_graph.findings:
        return "medium"
    return "low"


def _action_type(action: str) -> ActionType:
    lowered = action.lower()
    if lowered in {"profile_data", "advise_labels", "import_metrics", "mine_errors", "benchmark_latency"}:
        return lowered  # type: ignore[return-value]
    if any(token in lowered for token in ("oversampling", "hard_negative", "sampling", "background_only")):
        return "data_action"
    if any(token in lowered for token in ("label", "annotation", "audit")):
        return "label_action"
    if any(token in lowered for token in ("mosaic", "copy_paste", "crop", "blur", "noise")):
        return "augmentation_action"
    if any(token in lowered for token in ("sahi", "nms", "threshold", "tta", "slicing")):
        return "postprocess_action"
    return "run_training"


def _target_for_action(action: str, focus_items: list[dict[str, Any]]) -> str | None:
    for item in focus_items:
        actions = item.get("action_candidates", [])
        if isinstance(actions, list) and action in {str(value) for value in actions}:
            return str(item.get("class_name") or item.get("class_pair") or item.get("area") or item.get("subject"))
    if focus_items:
        item = focus_items[0]
        return str(item.get("class_name") or item.get("class_pair") or item.get("area") or item.get("subject"))
    return None


def _why_for_action(action: str, focus_items: list[dict[str, Any]]) -> list[str]:
    reasons = []
    for item in focus_items:
        actions = item.get("action_candidates", [])
        if isinstance(actions, list) and action in {str(value) for value in actions}:
            reason = item.get("reason")
            if reason:
                reasons.append(str(reason))
    return dedupe_list(reasons) or ["Selected by current error facts and loop policy."]


def _evidence_actions(missing_evidence: list[str]) -> dict[str, list[str]]:
    actions: dict[str, list[str]] = {}
    for name in missing_evidence:
        action = _evidence_action_for(name)
        actions.setdefault(action, []).append(name)
    return {action: dedupe_list(names) for action, names in actions.items()}


def _evidence_action_for(name: str) -> str:
    normalized = name.lower()
    if normalized in {"dataset_report", "dataset_health", "data_profile"}:
        return "profile_data"
    if normalized in {"label_quality_report", "label_quality_score"} or "label" in normalized:
        return "advise_labels"
    if normalized in {"latency", "latency_ms", "fps", "model_size_mb", "model_size"}:
        return "benchmark_latency"
    if normalized in {
        "confusion_matrix",
        "class_confusion_pairs",
        "false_positive_samples",
        "false_negative_samples",
        "false_positive_count",
        "false_negative_count",
        "localization_error_rate",
    }:
        return "mine_errors"
    return "import_metrics"


def _metric_label(metric: str) -> str:
    aliases = {
        "ap_small": "AP_small",
        "ap_medium": "AP_medium",
        "ap_large": "AP_large",
        "map50_95": "mAP50-95",
        "coco_ap50_95": "COCO mAP50-95",
        "per_class_ap": "per-class AP",
        "per_class_ar": "per-class AR",
    }
    return aliases.get(metric, metric)


def _metric_value(value: Any) -> float | int | str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    return None


def _dedupe_rejected(items: list[RejectedAction]) -> list[RejectedAction]:
    seen: set[tuple[str, str]] = set()
    output: list[RejectedAction] = []
    for item in items:
        key = (item.action, item.reason)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _dedupe_selected(items: list[SelectedAction]) -> list[SelectedAction]:
    seen: set[tuple[str, str | None]] = set()
    output: list[SelectedAction] = []
    for item in items:
        key = (item.action, item.target)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _grounding_catalog(
    rule_report: DoctorDecisionReport,
    evidence: Evidence,
    error_facts: list[ErrorFact],
) -> list[str]:
    entries: list[str] = []
    entries.extend(rule_report.evidence)
    entries.extend(_focus_evidence_line(item) for item in rule_report.target_error_facts)
    for fact in error_facts:
        entries.append(_error_fact_grounding_line(fact))
    for name, value in evidence.metrics.items():
        if value is not None:
            entries.append(f"{_metric_label(name)}={_metric_value(value)}")
            entries.append(f"{name}={_metric_value(value)}")
    for record in evidence.metric_records:
        if record.verified and record.value is not None:
            value = _metric_value(record.value)
            entries.append(
                " ".join(
                    str(item)
                    for item in [
                        record.metric_name,
                        _metric_label(record.metric_name),
                        value,
                        record.candidate_id,
                        record.node_id,
                        record.split,
                        record.validator,
                    ]
                    if item
                )
            )
    for artifact in evidence.artifact_manifest:
        if artifact.verify():
            entries.append(f"{artifact.name} {artifact.producer_stage} {artifact.type} {artifact.path.name}")
    return [entry for entry in dedupe_list(entries) if entry]


def _error_fact_grounding_line(fact: ErrorFact) -> str:
    value = _metric_value(fact.value)
    parts = [
        fact.fact_type,
        fact.subject,
        fact.class_name,
        fact.class_pair,
        fact.area,
        fact.metric_name,
        _metric_label(fact.metric_name or fact.fact_type),
        value,
        fact.count,
        fact.severity,
        fact.candidate_id,
        fact.node_id,
        fact.source,
    ]
    return " ".join(str(part) for part in parts if part is not None)


def _grounded_lines(raw_lines: Any, catalog: list[str]) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    for line in _string_list(raw_lines):
        if _line_is_grounded(line, catalog):
            accepted.append(line)
        else:
            rejected.append(line)
    return dedupe_list(accepted), dedupe_list(rejected)


def _line_is_grounded(line: str, catalog: list[str]) -> bool:
    line_tokens = _tokens(line)
    if not line_tokens:
        return False
    line_numbers = _number_tokens(line)
    for entry in catalog:
        entry_tokens = _tokens(entry)
        if not entry_tokens:
            continue
        overlap = line_tokens.intersection(entry_tokens)
        entry_numbers = _number_tokens(entry)
        if line_numbers and entry_numbers:
            if line_numbers.intersection(entry_numbers) and len(overlap - line_numbers) >= 1:
                return True
            continue
        if len(overlap) >= min(3, len(line_tokens)):
            return True
        if len(line_tokens) <= 2 and line_tokens.issubset(entry_tokens):
            return True
    return False


def _grounded_explanations(raw_lines: Any, grounded_context: set[str]) -> tuple[list[str], list[str]]:
    accepted: list[str] = []
    rejected: list[str] = []
    for line in _string_list(raw_lines):
        tokens = _tokens(line)
        if tokens and len(tokens.intersection(grounded_context)) >= 2:
            accepted.append(line)
        else:
            rejected.append(line)
    return dedupe_list(accepted), dedupe_list(rejected)


def _grounded_selected_actions(
    raw_actions: Any,
    rule_report: DoctorDecisionReport,
    error_facts: list[ErrorFact],
) -> tuple[list[SelectedAction], list[str]]:
    allowed = {item.action for item in rule_report.selected_actions}
    for fact in error_facts:
        allowed.update(fact.action_candidates)
    accepted: list[SelectedAction] = []
    rejected: list[str] = []
    for action in _string_list(raw_actions):
        if action in allowed:
            accepted.append(
                SelectedAction(
                    action=action,
                    action_type=_action_type(action),
                    target=_target_for_action(action, rule_report.target_error_facts),
                    why=["LLM draft action accepted because it matches grounded error-fact actions."],
                )
            )
        else:
            rejected.append(action)
    return _dedupe_selected(accepted), dedupe_list(rejected)


def _grounded_rejected_actions(
    raw_actions: Any,
    rule_report: DoctorDecisionReport,
) -> tuple[list[RejectedAction], list[str]]:
    allowed = {item.action for item in rule_report.rejected_actions}
    accepted: list[RejectedAction] = []
    rejected: list[str] = []
    raw_items = raw_actions if isinstance(raw_actions, list) else []
    for item in raw_items:
        if not isinstance(item, dict):
            rejected.append(str(item))
            continue
        action = str(item.get("action", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if action and action in allowed and reason:
            accepted.append(RejectedAction(action=action, reason=reason))
        elif action:
            rejected.append(action)
    return _dedupe_rejected(accepted), dedupe_list(rejected)


def _grounded_expected_improvement(
    raw_expected: Any,
    grounded_context: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(raw_expected, dict):
        return {}, {}
    accepted: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for key, value in raw_expected.items():
        metric = str(key)
        text = str(value)
        if _tokens(metric).intersection(grounded_context):
            accepted[metric] = text
        else:
            rejected[metric] = text
    return accepted, rejected


def _catalog_tokens(catalog: list[str]) -> set[str]:
    tokens: set[str] = set()
    for entry in catalog:
        tokens.update(_tokens(entry))
    return tokens


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _tokens(text: str) -> set[str]:
    normalized = text.replace("-", "_")
    return {
        token
        for token in re.findall(r"[a-zA-Z_]+|\d+(?:\.\d+)?", normalized.lower())
        if token not in {"the", "and", "for", "with", "from", "this", "that", "into", "before", "after"}
    }


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.lower()))
