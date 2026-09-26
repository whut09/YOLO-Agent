"""Structured error facts for closed-loop detection optimization."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, field_serializer

from yolo_agent.core.experiment_graph import MetricValue


ErrorFactType = Literal[
    "per_class_metric",
    "area_metric",
    "class_low_ap",
    "false_negative_heavy_class",
    "localization_heavy_class",
    "class_confusion_pair",
    "background_false_positive_class",
    "subset_performance",
    "high_confidence_false_positive",
    "confidence_localization_mismatch",
    "localization_error",
    "assignment_conflict",
    "duplicate_prediction",
    "scale_variation",
    "feature_relation_gap",
    "representation_gap",
    "capacity_gap",
    # Prompt-15 vocabulary: global, confidence, slice, and explicit
    # evidence-incomplete facts (missing evidence is recorded, never guessed).
    "global_metric",
    "confidence_calibration",
    "slice_performance",
    "evidence_incomplete",
]

ErrorSeverity = Literal["low", "medium", "high"]


class ErrorFact(BaseModel):
    """One queryable error fact tied to a candidate and experiment node."""

    run_id: str
    origin_run_id: str | None = None
    inheritance_depth: int = Field(default=0, ge=0)
    candidate_id: str
    node_id: str
    dataset_version: str = "unversioned"
    dataset_manifest_sha256: str | None = None
    subset_manifest_sha256: str | None = None
    split: str = "val"
    protocol_hash: str | None = None
    seed: int | str | None = None
    fidelity: str | None = None
    epochs: int | None = Field(default=None, ge=1)
    batch_policy_hash: str | None = None
    ultralytics_version: str | None = None
    imgsz: int | None = Field(default=None, ge=1)
    eval_protocol_hash: str | None = None
    evidence_role: Literal["current_observation", "inherited_context", "baseline_reference"] = "current_observation"
    fact_type: ErrorFactType
    subject: str
    class_name: str | None = None
    class_pair: str | None = None
    area: str | None = None
    metric_name: str | None = None
    value: MetricValue = None
    count: int | None = None
    rank: int | None = None
    severity: ErrorSeverity = "medium"
    evidence: dict[str, MetricValue] = Field(default_factory=dict)
    action_candidates: list[str] = Field(default_factory=list)
    source: str = "manual"
    source_artifact: Path | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_serializer("source_artifact")
    def serialize_source_artifact(self, value: Path | None) -> str | None:
        """Serialize source artifact paths portably."""
        return value.as_posix() if value is not None else None


class ErrorFactQuery(BaseModel):
    """Filters for error fact lookup."""

    candidate_id: str | None = None
    node_id: str | None = None
    dataset_version: str | None = None
    split: str | None = None
    fact_type: ErrorFactType | None = None
    subject: str | None = None
    class_name: str | None = None
    area: str | None = None
    severity: ErrorSeverity | None = None


class ErrorFactIndex:
    """Queryable index over error facts."""

    def __init__(self, facts: Iterable[ErrorFact]) -> None:
        self.facts = list(facts)

    def query(
        self,
        candidate_id: str | None = None,
        node_id: str | None = None,
        dataset_version: str | None = None,
        split: str | None = None,
        fact_type: ErrorFactType | None = None,
        subject: str | None = None,
        class_name: str | None = None,
        area: str | None = None,
        severity: ErrorSeverity | None = None,
    ) -> list[ErrorFact]:
        """Return facts matching all supplied filters."""
        query = ErrorFactQuery(
            candidate_id=candidate_id,
            node_id=node_id,
            dataset_version=dataset_version,
            split=split,
            fact_type=fact_type,
            subject=subject,
            class_name=class_name,
            area=area,
            severity=severity,
        )
        return [fact for fact in self.facts if _matches(fact, query)]

    def action_candidates(
        self,
        candidate_id: str | None = None,
        node_id: str | None = None,
        severity: ErrorSeverity | None = None,
    ) -> list[str]:
        """Return deduplicated action candidates for matching facts."""
        actions: list[str] = []
        for fact in self.query(candidate_id=candidate_id, node_id=node_id, severity=severity):
            actions.extend(fact.action_candidates)
        return list(dict.fromkeys(actions))


_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def _replace_with_retry(temp_path: Path, path: Path) -> None:
    """Atomically replace ``path`` with ``temp_path``, tolerating Windows locks.

    On Windows a transient external reader (indexer, antivirus scan, or a
    concurrent heartbeat reader) can hold the destination open while
    ``os.replace`` runs, surfacing as ``PermissionError`` (WinError 5).
    The lock clears within milliseconds, so retry with backoff before
    failing closed; the staged ``.tmp`` file keeps every attempt lossless.
    """
    for delay in (*_REPLACE_RETRY_DELAYS, None):
        try:
            temp_path.replace(path)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


class ErrorFactStore:
    """Append-only error fact JSONL storage under runs/{run_id}."""

    def __init__(self, root: Path | str = "runs") -> None:
        self.root = Path(root)

    def append(self, run_id: str, facts: list[ErrorFact]) -> Path:
        """Append error facts to ``error_facts_by_node.jsonl``."""
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "error_facts_by_node.jsonl"
        with path.open("a", encoding="utf-8") as file:
            for fact in facts:
                file.write(json.dumps(fact.model_dump(mode="json"), sort_keys=True) + "\n")
        return path

    def replace_current_node(
        self,
        run_id: str,
        candidate_id: str,
        node_id: str,
        protocol_hash: str,
        facts: list[ErrorFact],
        *,
        evidence_role: Literal["current_observation", "baseline_reference"] = "current_observation",
    ) -> Path:
        """Atomically replace one node's facts for an exact protocol and role."""
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "error_facts_by_node.jsonl"
        retained = [
            fact
            for fact in self.read(run_id)
            if not (
                fact.run_id == run_id
                and fact.candidate_id == candidate_id
                and fact.node_id == node_id
                and fact.protocol_hash == protocol_hash
                and fact.evidence_role == evidence_role
            )
        ]
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            for fact in [*retained, *facts]:
                file.write(json.dumps(fact.model_dump(mode="json"), sort_keys=True) + "\n")
        _replace_with_retry(temp_path, path)
        return path

    def read(self, run_id: str) -> list[ErrorFact]:
        """Read all persisted error facts for a run."""
        path = self._run_dir(run_id) / "error_facts_by_node.jsonl"
        if not path.is_file():
            return []
        facts: list[ErrorFact] = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                text = line.strip()
                if text:
                    facts.append(ErrorFact.model_validate(json.loads(text)))
        return facts

    def index(self, run_id: str) -> ErrorFactIndex:
        """Return a query index for one run."""
        return ErrorFactIndex(self.read(run_id))

    def _run_dir(self, run_id: str) -> Path:
        if not run_id or any(separator in run_id for separator in ("/", "\\")):
            raise ValueError("run_id must be a non-empty single path segment.")
        return self.root / run_id


def build_error_facts_from_coco_metrics(
    metrics: dict[str, MetricValue],
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "coco_eval_importer",
    source_artifact: Path | str | None = None,
    low_ap_threshold: float = 0.35,
    top_k_low_ap: int = 10,
) -> list[ErrorFact]:
    """Build standardized error facts from COCO metric evidence."""
    artifact = Path(source_artifact) if source_artifact is not None else None
    facts: list[ErrorFact] = []
    for name, value in metrics.items():
        numeric = _numeric(value)
        if name.startswith("per_class_ap/"):
            class_name = name.split("/", 1)[1]
            facts.append(
                _fact(
                    run_id,
                    candidate_id,
                    node_id,
                    dataset_version,
                    split,
                    fact_type="per_class_metric",
                    subject=class_name,
                    class_name=class_name,
                    metric_name="per_class_ap",
                    value=value,
                    severity=_severity_for_score(numeric, low_ap_threshold),
                    actions=["inspect_class_errors", "audit_labels_for_class"],
                    source=source,
                    artifact=artifact,
                )
            )
        elif name.startswith("per_class_ar/"):
            class_name = name.split("/", 1)[1]
            facts.append(
                _fact(
                    run_id,
                    candidate_id,
                    node_id,
                    dataset_version,
                    split,
                    fact_type="per_class_metric",
                    subject=class_name,
                    class_name=class_name,
                    metric_name="per_class_ar",
                    value=value,
                    severity=_severity_for_score(numeric, low_ap_threshold),
                    actions=["inspect_false_negatives", "class_balanced_sampling"],
                    source=source,
                    artifact=artifact,
                )
            )
        elif name in {"ap_small", "ap_medium", "ap_large", "ar_small", "ar_medium", "ar_large"}:
            area = name.rsplit("_", 1)[1]
            actions = _area_actions(name, numeric, low_ap_threshold)
            facts.append(
                _fact(
                    run_id,
                    candidate_id,
                    node_id,
                    dataset_version,
                    split,
                    fact_type="area_metric",
                    subject=area,
                    area=area,
                    metric_name=name,
                    value=value,
                    severity=_severity_for_score(numeric, low_ap_threshold),
                    actions=actions,
                    source=source,
                    artifact=artifact,
                )
            )
    low_ap = sorted(
        [
            (name.split("/", 1)[1], _numeric(value))
            for name, value in metrics.items()
            if name.startswith("per_class_ap/") and _numeric(value) is not None
        ],
        key=lambda item: item[1] if item[1] is not None else 1.0,
    )[:top_k_low_ap]
    for rank, (class_name, value) in enumerate(low_ap, start=1):
        if value is None or value > low_ap_threshold:
            continue
        facts.append(
            _fact(
                run_id,
                candidate_id,
                node_id,
                dataset_version,
                split,
                fact_type="class_low_ap",
                subject=class_name,
                class_name=class_name,
                metric_name="per_class_ap",
                value=value,
                rank=rank,
                severity=_severity_for_score(value, low_ap_threshold),
                actions=["collect_more_samples_for_class", "audit_labels_for_class", "class_balanced_sampling"],
                source=source,
                artifact=artifact,
            )
        )
    return facts


def build_error_facts_from_coco_error_report(
    report: dict[str, Any],
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "coco_error_mining",
    source_artifact: Path | str | None = None,
) -> list[ErrorFact]:
    """Build facts from a ``CocoErrorReport`` JSON mapping."""
    artifact = Path(source_artifact) if source_artifact is not None else None
    facts: list[ErrorFact] = []
    facts.extend(
        _class_count_facts(
            report.get("false_negative_top_classes", []),
            run_id,
            candidate_id,
            node_id,
            dataset_version,
            split,
            fact_type="false_negative_heavy_class",
            count_key="false_negative",
            actions=["audit_label_noise", "increase_recall_recipe", "class_balanced_sampling"],
            source=source,
            artifact=artifact,
        )
    )
    facts.extend(
        _class_count_facts(
            report.get("localization_error_top_classes", []),
            run_id,
            candidate_id,
            node_id,
            dataset_version,
            split,
            fact_type="localization_heavy_class",
            count_key="localization_error",
            actions=["bbox_loss_recipe", "assigner_recipe", "label_box_audit"],
            source=source,
            artifact=artifact,
        )
    )
    facts.extend(
        _class_count_facts(
            report.get("background_false_positive_top_classes", []),
            run_id,
            candidate_id,
            node_id,
            dataset_version,
            split,
            fact_type="background_false_positive_class",
            count_key="background_false_positive",
            actions=["hard_negative_mining", "background_only_sampling", "precision_threshold_tuning"],
            source=source,
            artifact=artifact,
        )
    )
    pairs = report.get("class_confusion_pairs", {})
    if isinstance(pairs, dict):
        for rank, (pair, count) in enumerate(sorted(pairs.items(), key=lambda item: int(item[1]), reverse=True), start=1):
            facts.append(
                _fact(
                    run_id,
                    candidate_id,
                    node_id,
                    dataset_version,
                    split,
                    fact_type="class_confusion_pair",
                    subject=str(pair),
                    class_pair=str(pair),
                    count=int(count),
                    rank=rank,
                    severity=_severity_for_count(int(count)),
                    actions=["class_definition_audit", "confusion_pair_sampling", "classification_loss_recipe"],
                    source=source,
                    artifact=artifact,
                )
            )
    for area, value in _mapping(report.get("area_ap50")).items():
        facts.append(
            _fact(
                run_id,
                candidate_id,
                node_id,
                dataset_version,
                split,
                fact_type="subset_performance",
                subject=f"{area}_ap50",
                area=area,
                metric_name="area_ap50",
                value=value,
                severity=_severity_for_score(_numeric(value), 0.35),
                actions=_area_actions(f"ap_{area}", _numeric(value), 0.35),
                source=source,
                artifact=artifact,
            )
        )
    for area, value in _mapping(report.get("area_recall")).items():
        facts.append(
            _fact(
                run_id,
                candidate_id,
                node_id,
                dataset_version,
                split,
                fact_type="subset_performance",
                subject=f"{area}_recall",
                area=area,
                metric_name="area_recall",
                value=value,
                severity=_severity_for_score(_numeric(value), 0.5),
                actions=_area_actions(f"ar_{area}", _numeric(value), 0.5),
                source=source,
                artifact=artifact,
            )
        )
    return facts


GLOBAL_FACT_METRICS = ("map50", "map50_95", "precision", "recall")


def build_error_facts_from_global_metrics(
    metrics: dict[str, MetricValue],
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "coco_eval_importer",
    source_artifact: Path | str | None = None,
) -> list[ErrorFact]:
    """Build global-metric facts (mAP50, mAP50-95, precision, recall)."""

    artifact = Path(source_artifact) if source_artifact is not None else None
    facts: list[ErrorFact] = []
    for name in GLOBAL_FACT_METRICS:
        if name not in metrics:
            continue
        value = metrics[name]
        facts.append(
            _fact(
                run_id,
                candidate_id,
                node_id,
                dataset_version,
                split,
                fact_type="global_metric",
                subject=name,
                metric_name=name,
                value=value,
                severity=_severity_for_score(_numeric(value), 0.35),
                actions=["import_metrics"],
                source=source,
                artifact=artifact,
            )
        )
    return facts


def build_error_facts_from_confidence_profile(
    profile: dict[str, MetricValue],
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "confidence_profiler",
    source_artifact: Path | str | None = None,
) -> list[ErrorFact]:
    """Build confidence facts: TP confidence, FP confidence, calibration (ECE).

    Accepted keys: ``tp_confidence``, ``fp_confidence``, ``ece``.  A high ECE
    or a large TP/FP confidence gap yields a ``confidence_calibration`` fact;
    nothing is inferred when keys are absent.
    """

    artifact = Path(source_artifact) if source_artifact is not None else None
    accepted = {"tp_confidence", "fp_confidence", "ece"}
    evidence = {key: _metric_value(profile[key]) for key in profile if key in accepted}
    if not evidence:
        return []
    ece = _numeric(evidence.get("ece"))
    tp = _numeric(evidence.get("tp_confidence"))
    fp = _numeric(evidence.get("fp_confidence"))
    severity: ErrorSeverity = "low"
    if ece is not None and ece >= 0.1:
        severity = "high"
    elif tp is not None and fp is not None and abs(tp - fp) < 0.15:
        severity = "high"
    return [
        _fact(
            run_id,
            candidate_id,
            node_id,
            dataset_version,
            split,
            fact_type="confidence_calibration",
            subject="confidence_profile",
            metric_name="ece" if ece is not None else "tp_confidence",
            value=evidence.get("ece", evidence.get("tp_confidence")),
            evidence=evidence,
            severity=severity,
            actions=["calibrate_scores", "tune_confidence_threshold"],
            source=source,
            artifact=artifact,
        )
    ]


def build_error_facts_from_slices(
    slices: dict[str, dict[str, MetricValue]],
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "slice_eval",
    source_artifact: Path | str | None = None,
    low_ap_threshold: float = 0.35,
) -> list[ErrorFact]:
    """Build per-slice facts (scene/domain/class/scale sub-metrics).

    Each slice entry maps ``{"map50_95": value, ...}``; slice keys are
    prefixed so queries can target scene/domain/class/scale explicitly.
    """

    artifact = Path(source_artifact) if source_artifact is not None else None
    facts: list[ErrorFact] = []
    for slice_key, metrics in slices.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, value in metrics.items():
            numeric = _numeric(value)
            if numeric is None:
                continue
            facts.append(
                _fact(
                    run_id,
                    candidate_id,
                    node_id,
                    dataset_version,
                    split,
                    fact_type="slice_performance",
                    subject=f"{slice_key}/{metric_name}",
                    metric_name=f"slice/{slice_key}/{metric_name}",
                    value=numeric,
                    severity=_severity_for_score(numeric, low_ap_threshold),
                    actions=["review_slice_errors", "domain_check"],
                    source=source,
                    artifact=artifact,
                )
            )
    return facts


def build_evidence_incomplete_fact(
    run_id: str,
    candidate_id: str,
    node_id: str,
    missing_metrics: list[str],
    dataset_version: str = "coco2017",
    split: str = "val2017",
    source: str = "evidence_gap_scan",
) -> ErrorFact:
    """Explicitly record that required evidence is missing.

    The fact carries the missing metric names in ``evidence`` with sentinel
    ``"missing"`` values.  Downstream consumers must treat these as unknown —
    never interpolate a number — and plan collection instead.
    """

    return _fact(
        run_id,
        candidate_id,
        node_id,
        dataset_version,
        split,
        fact_type="evidence_incomplete",
        subject="missing_metrics",
        metric_name="evidence_incomplete",
        value=None,
        evidence={name: "missing" for name in missing_metrics},
        severity="high",
        actions=["collect_evidence"],
        source=source,
        artifact=None,
    )


def _class_count_facts(
    items: Any,
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str,
    split: str,
    fact_type: ErrorFactType,
    count_key: str,
    actions: list[str],
    source: str,
    artifact: Path | None,
) -> list[ErrorFact]:
    if not isinstance(items, list):
        return []
    facts: list[ErrorFact] = []
    for rank, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        count = int(item.get(count_key, 0) or 0)
        if count <= 0:
            continue
        class_name = str(item.get("name") or item.get("class_name") or item.get("class") or item.get("category_id"))
        facts.append(
            _fact(
                run_id,
                candidate_id,
                node_id,
                dataset_version,
                split,
                fact_type=fact_type,
                subject=class_name,
                class_name=class_name,
                count=count,
                rank=rank,
                severity=_severity_for_count(count),
                evidence={key: _metric_value(value) for key, value in item.items() if key != "name"},
                actions=actions,
                source=source,
                artifact=artifact,
            )
        )
    return facts


def _fact(
    run_id: str,
    candidate_id: str,
    node_id: str,
    dataset_version: str,
    split: str,
    fact_type: ErrorFactType,
    subject: str,
    severity: ErrorSeverity,
    actions: list[str],
    source: str,
    artifact: Path | None,
    class_name: str | None = None,
    class_pair: str | None = None,
    area: str | None = None,
    metric_name: str | None = None,
    value: MetricValue = None,
    count: int | None = None,
    rank: int | None = None,
    evidence: dict[str, MetricValue] | None = None,
) -> ErrorFact:
    return ErrorFact(
        run_id=run_id,
        origin_run_id=run_id,
        candidate_id=candidate_id,
        node_id=node_id,
        dataset_version=dataset_version,
        split=split,
        fact_type=fact_type,
        subject=subject,
        class_name=class_name,
        class_pair=class_pair,
        area=area,
        metric_name=metric_name,
        value=value,
        count=count,
        rank=rank,
        severity=severity,
        evidence=evidence or {},
        action_candidates=actions,
        source=source,
        source_artifact=artifact,
    )


def _matches(fact: ErrorFact, query: ErrorFactQuery) -> bool:
    return all(
        [
            query.candidate_id is None or fact.candidate_id == query.candidate_id,
            query.node_id is None or fact.node_id == query.node_id,
            query.dataset_version is None or fact.dataset_version == query.dataset_version,
            query.split is None or fact.split == query.split,
            query.fact_type is None or fact.fact_type == query.fact_type,
            query.subject is None or fact.subject == query.subject,
            query.class_name is None or fact.class_name == query.class_name,
            query.area is None or fact.area == query.area,
            query.severity is None or fact.severity == query.severity,
        ]
    )


def _area_actions(name: str, value: float | None, threshold: float) -> list[str]:
    if value is not None and value > threshold:
        return ["monitor_subset_performance"]
    if "small" in name:
        return ["small_object_recipe", "sahi_or_tiling_eval", "copy_paste_small_objects"]
    return ["subset_error_review", "sampling_policy_review"]


def _severity_for_score(value: float | None, threshold: float) -> ErrorSeverity:
    if value is None:
        return "medium"
    if value <= threshold * 0.6:
        return "high"
    if value <= threshold:
        return "medium"
    return "low"


def _severity_for_count(count: int) -> ErrorSeverity:
    if count >= 50:
        return "high"
    if count >= 10:
        return "medium"
    return "low"


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _metric_value(value: Any) -> MetricValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _mapping(value: Any) -> dict[str, MetricValue]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _metric_value(raw) for key, raw in value.items()}
