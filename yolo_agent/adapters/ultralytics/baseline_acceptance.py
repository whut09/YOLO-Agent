"""COCO baseline acceptance gate for full-budget candidate promotion."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from yolo_agent.core.artifact_manifest import ArtifactManifestEntry
from yolo_agent.core.coco_baseline_evidence import CocoBaselineEvidenceContract
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.experiment_graph import Evidence, MetricEvidence, MetricValue


class BaselineAcceptanceConfig(BaseModel):
    """Protocol for trusting a COCO baseline before running full candidates."""

    enabled: bool = True
    required_metric_candidates: list[str] = Field(default_factory=lambda: ["map50_95", "coco_ap50_95"])
    required_artifacts: list[str] = Field(default_factory=lambda: ["results_csv", "best_pt", "args_yaml"])
    required_imgsz: int = Field(default=640, ge=1)
    allowed_profiles: list[str] = Field(default_factory=lambda: ["baseline_full", "baseline_confirm"])
    minimum_seeds: int = Field(default=3, ge=1)
    expected_dataset_manifest_sha256: str | None = None
    require_dataset_manifest_match: bool = True
    allow_explained_bottleneck: bool = True
    enforce_coco_baseline_evidence: bool = True
    severe_bottleneck_severity: str = "high"
    require_artifact_provenance: bool = False
    preferred_metric_validators: list[str] = Field(
        default_factory=lambda: [
            "ultralytics_results_importer",
            "coco_error_importer",
            "official_eval",
            "benchmark_import",
        ]
    )


class BaselineAcceptanceResult(BaseModel):
    """Decision from the COCO baseline acceptance gate."""

    baseline_trusted: bool
    baseline_rejection_reason: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    accepted_seed_count: int = 0
    accepted_nodes: list[str] = Field(default_factory=list)
    accepted_candidates: list[str] = Field(default_factory=list)
    required_imgsz: int = 640
    expected_dataset_manifest_sha256: str | None = None
    actual_dataset_manifest_sha256: str | None = None
    expected_protocol_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BaselineAcceptanceGate:
    """Check that a full COCO baseline is trusted before candidate_full."""

    def __init__(self, config: BaselineAcceptanceConfig | None = None) -> None:
        self.config = config or BaselineAcceptanceConfig()

    def check(
        self,
        evidence: Evidence,
        expected_dataset_manifest_sha256: str | None = None,
        actual_dataset_manifest_sha256: str | None = None,
        expected_protocol_hash: str | None = None,
    ) -> BaselineAcceptanceResult:
        """Return whether current run evidence satisfies the baseline protocol."""
        if not self.config.enabled:
            return BaselineAcceptanceResult(
                baseline_trusted=True,
                warnings=["Baseline acceptance gate is disabled."],
                required_imgsz=self.config.required_imgsz,
            )

        reasons: list[str] = []
        warnings: list[str] = []
        expected_sha = expected_dataset_manifest_sha256 or self.config.expected_dataset_manifest_sha256
        actual_sha = actual_dataset_manifest_sha256
        if self.config.require_dataset_manifest_match:
            if expected_sha is None:
                reasons.append("missing_expected_coco_manifest_sha256")
            if actual_sha is None:
                reasons.append("missing_dataset_manifest_sha256")
            if expected_sha is not None and actual_sha is not None and actual_sha != expected_sha:
                reasons.append("dataset_manifest_sha256_mismatch")

        node_ids = _profile_node_ids(evidence, self.config.allowed_profiles)
        if not node_ids:
            reasons.append(f"missing_allowed_profile:{','.join(self.config.allowed_profiles)}")

        accepted_nodes: list[str] = []
        accepted_candidates: list[str] = []
        accepted_seeds: set[str] = set()
        for node_id in sorted(node_ids):
            node_reasons = _node_rejection_reasons(
                node_id,
                evidence,
                self.config,
                expected_protocol_hash=expected_protocol_hash,
            )
            if node_reasons:
                reasons.extend(f"{node_id}:{reason}" for reason in node_reasons)
                continue
            accepted_nodes.append(node_id)
            candidate_id = _candidate_id_for_node(evidence.metric_records, node_id)
            if candidate_id is not None:
                accepted_candidates.append(candidate_id)
            seed = _seed_for_node(evidence.metric_records, node_id, evidence.run_id, expected_protocol_hash)
            if seed is not None:
                accepted_seeds.add(seed)

        if len(accepted_seeds) < self.config.minimum_seeds:
            reasons.append(
                f"insufficient_confirmed_seeds:{len(accepted_seeds)}/{self.config.minimum_seeds}"
            )
        if self.config.enforce_coco_baseline_evidence:
            contract = CocoBaselineEvidenceContract().evaluate(evidence)
            if not contract.ok:
                reasons.extend(f"coco_baseline_evidence:{item}" for item in contract.missing_required)

        trusted = not reasons
        return BaselineAcceptanceResult(
            baseline_trusted=trusted,
            baseline_rejection_reason=list(dict.fromkeys(reasons)),
            warnings=list(dict.fromkeys(warnings)),
            accepted_seed_count=len(accepted_seeds),
            accepted_nodes=accepted_nodes,
            accepted_candidates=list(dict.fromkeys(accepted_candidates)),
            required_imgsz=self.config.required_imgsz,
            expected_dataset_manifest_sha256=expected_sha,
            actual_dataset_manifest_sha256=actual_sha,
            expected_protocol_hash=expected_protocol_hash,
        )

    def persist_decision(
        self,
        store: EvidenceStore,
        run_id: str,
        result: BaselineAcceptanceResult,
        dataset_version: str = "unversioned",
    ) -> Path:
        """Persist the gate result as an artifact plus queryable metrics."""
        artifact_path = store.create_run(run_id) / "artifacts" / "baseline_acceptance.json"
        artifact_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        store.log_artifact_manifest(
            run_id=run_id,
            name="baseline_acceptance",
            artifact_path=artifact_path,
            producer_stage="baseline_acceptance_gate",
        )
        metrics = _current_run_metrics(store, run_id)
        metrics.update(
            {
                "baseline_trusted": result.baseline_trusted,
                "baseline_rejection_reason": ";".join(result.baseline_rejection_reason),
                "baseline_accepted_seed_count": result.accepted_seed_count,
            }
        )
        store.log_metrics(run_id, metrics)
        store.log_candidate_metrics(
            run_id=run_id,
            candidate_id="baseline_acceptance",
            node_id="baseline_acceptance",
            metrics={
                "baseline_trusted": result.baseline_trusted,
                "baseline_rejection_reason": ";".join(result.baseline_rejection_reason),
                "baseline_accepted_seed_count": result.accepted_seed_count,
            },
            dataset_version=dataset_version,
            split="runtime",
            source="baseline_acceptance_gate",
            verified=True,
            validator="baseline_acceptance_gate",
            source_artifact=artifact_path,
        )
        return artifact_path


def _profile_node_ids(evidence: Evidence, allowed_profiles: list[str]) -> set[str]:
    allowed = set(allowed_profiles)
    node_ids: set[str] = set()
    for record in evidence.metric_records:
        if record.metric_name in {"training_budget_profile", "fast_baseline_gate_profile"}:
            if (
                str(record.value) in allowed
                and record.verified
                and record.run_id == evidence.run_id
                and record.origin_run_id in {None, evidence.run_id}
                and record.evidence_role == "current_observation"
                and record.inheritance_depth == 0
                and not record.source.startswith("inherited:")
            ):
                node_ids.add(record.node_id)
    return node_ids


def _node_rejection_reasons(
    node_id: str,
    evidence: Evidence,
    config: BaselineAcceptanceConfig,
    *,
    expected_protocol_hash: str | None,
) -> list[str]:
    reasons: list[str] = []
    if _candidate_id_for_node(evidence.metric_records, node_id) is None:
        reasons.append("missing_candidate_id")
    if _seed_for_node(evidence.metric_records, node_id, evidence.run_id, expected_protocol_hash) is None:
        reasons.append("missing_seed_evidence")
    metric_record = _metric_record_for_node(
        evidence,
        node_id,
        config,
        expected_protocol_hash=expected_protocol_hash,
    )
    if metric_record is None:
        reasons.append(f"missing_verified_metric:{'/'.join(config.required_metric_candidates)}")

    artifact_entries = {
        artifact_name: _artifact_entry_for_node(
            evidence.artifact_manifest,
            evidence.run_id,
            node_id,
            artifact_name,
            expected_protocol_hash or (metric_record.protocol_hash if metric_record is not None else None),
            require_provenance=config.require_artifact_provenance,
        )
        for artifact_name in config.required_artifacts
    }
    for artifact_name, entry in artifact_entries.items():
        if entry is None:
            reasons.append(f"missing_artifact:{artifact_name}")
        elif not entry.verify():
            reasons.append(f"stale_artifact:{artifact_name}")

    args_entry = artifact_entries.get("args_yaml")
    if args_entry is not None and args_entry.verify():
        imgsz = _imgsz_from_args(args_entry.path)
        if imgsz is None:
            reasons.append("missing_imgsz_in_args_yaml")
        elif imgsz != config.required_imgsz:
            reasons.append(f"imgsz_mismatch:{imgsz}!={config.required_imgsz}")

    bottleneck_reason = _severe_runtime_bottleneck_reason(evidence.metric_records, node_id, config)
    if bottleneck_reason:
        reasons.append(bottleneck_reason)
    return reasons


def _metric_record_for_node(
    evidence: Evidence,
    node_id: str,
    config: BaselineAcceptanceConfig,
    *,
    expected_protocol_hash: str | None,
) -> MetricEvidence | None:
    for metric_name in config.required_metric_candidates:
        candidates = [
            record
            for record in evidence.metric_records
            if record.node_id == node_id
            and record.metric_name == metric_name
            and record.split == "val"
            and record.verified
            and record.run_id == evidence.run_id
            and record.origin_run_id in {None, evidence.run_id}
            and record.evidence_role == "current_observation"
            and record.inheritance_depth == 0
            and not record.source.startswith("inherited:")
            and (expected_protocol_hash is None or record.protocol_hash == expected_protocol_hash)
            and _numeric(record.value) is not None
        ]
        if candidates:
            preferred = [r for r in candidates if r.validator in config.preferred_metric_validators]
            return max(preferred or candidates, key=lambda record: record.created_at)
    return None


def _artifact_entry_for_node(
    entries: list[ArtifactManifestEntry],
    run_id: str,
    node_id: str,
    artifact_name: str,
    expected_protocol_hash: str | None,
    *,
    require_provenance: bool,
) -> ArtifactManifestEntry | None:
    strict_candidates = [
        entry
        for entry in entries
        if entry.name == f"{node_id}_{artifact_name}"
        and entry.run_id == run_id
        and entry.node_id == node_id
        and (expected_protocol_hash is None or entry.protocol_hash == expected_protocol_hash)
    ]
    candidates = strict_candidates
    if not candidates and not require_provenance:
        aliases = {f"{node_id}_{artifact_name}", artifact_name}
        candidates = [entry for entry in entries if entry.name in aliases]
    if not candidates:
        return None
    verified = [entry for entry in candidates if entry.verify()]
    return max(verified or candidates, key=lambda entry: entry.created_at)


def _imgsz_from_args(path: Path) -> int | None:
    with path.open("r", encoding="utf-8-sig") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict) or "imgsz" not in data:
        return None
    value = data["imgsz"]
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _severe_runtime_bottleneck_reason(
    records: list[MetricEvidence],
    node_id: str,
    config: BaselineAcceptanceConfig,
) -> str | None:
    bottleneck = _record_value(records, node_id, "runtime_bottleneck")
    if bottleneck is not True:
        return None
    severity = str(_record_value(records, node_id, "runtime_bottleneck_severity") or "").lower()
    if severity != config.severe_bottleneck_severity.lower():
        return None
    if not config.allow_explained_bottleneck:
        return "severe_runtime_bottleneck_unexplained"
    explained = _record_value(records, node_id, "runtime_bottleneck_explained")
    explanation = _record_value(records, node_id, "runtime_bottleneck_explanation")
    if explained is True or (isinstance(explanation, str) and explanation.strip()):
        return None
    return "severe_runtime_bottleneck_unexplained"


def _candidate_id_for_node(records: list[MetricEvidence], node_id: str) -> str | None:
    for record in records:
        if record.node_id == node_id and record.candidate_id:
            return record.candidate_id
    return None


def _seed_for_node(
    records: list[MetricEvidence],
    node_id: str,
    run_id: str | None = None,
    expected_protocol_hash: str | None = None,
) -> str | None:
    candidates = [
        record
        for record in records
        if record.node_id == node_id
        and record.verified
        and (run_id is None or record.run_id == run_id)
        and record.origin_run_id in {None, run_id}
        and record.evidence_role == "current_observation"
        and record.inheritance_depth == 0
        and not record.source.startswith("inherited:")
        and (expected_protocol_hash is None or record.protocol_hash == expected_protocol_hash)
    ]
    explicit = [record for record in candidates if record.seed is not None]
    if explicit:
        return str(max(explicit, key=lambda record: record.created_at).seed)
    values = [record for record in candidates if record.metric_name == "fast_baseline_seed"]
    if not values:
        return None
    return str(max(values, key=lambda record: record.created_at).value)


def _record_value(records: list[MetricEvidence], node_id: str, metric_name: str) -> MetricValue:
    values = [
        record
        for record in records
        if record.node_id == node_id and record.metric_name == metric_name and record.verified
    ]
    if not values:
        return None
    return max(values, key=lambda record: record.created_at).value


def _numeric(value: MetricValue) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _current_run_metrics(store: EvidenceStore, run_id: str) -> dict[str, MetricValue]:
    try:
        return dict(store.load_run(run_id).metrics)
    except FileNotFoundError:
        return {}
