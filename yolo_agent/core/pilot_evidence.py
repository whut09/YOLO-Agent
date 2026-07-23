"""Completeness gate for candidate-specific pilot COCO evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore


class PilotEvidenceCompletenessResult(BaseModel):
    """Evidence readiness for one executed pilot node."""

    run_id: str
    candidate_id: str
    node_id: str
    protocol_hash: str
    complete: bool = False
    missing_metrics: list[str] = Field(default_factory=list)
    missing_artifacts: list[str] = Field(default_factory=list)
    invalid_artifacts: dict[str, list[str]] = Field(default_factory=dict)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    artifact_contract_hash: str | None = None
    missing_error_report_fields: list[str] = Field(default_factory=list)
    missing_fact_groups: list[str] = Field(default_factory=list)
    evidence_actions: list[str] = Field(default_factory=list)


class PilotEvidenceCompletenessGate:
    """Require current-node COCO evidence before another training proposal."""

    required_metrics = (
        "ap_small",
        "ap_medium",
        "ap_large",
        "fn_heavy_classes",
        "background_fp_classes",
        "localization_heavy_classes",
        "confusion_summary",
    )
    required_metric_prefixes = ("per_class_ap/", "per_class_ar/")
    required_artifacts = ("coco_predictions", "coco_eval", "coco_error_report")
    required_report_fields = (
        "false_negative_top_classes",
        "localization_error_top_classes",
        "background_false_positive_top_classes",
        "class_confusion_pairs",
    )
    required_fact_groups = ("area_metric", "per_class_metric")

    def __init__(self, evidence_store: EvidenceStore) -> None:
        self.evidence_store = evidence_store

    def evaluate(
        self,
        *,
        run_id: str,
        candidate_id: str,
        node_id: str,
        protocol_hash: str,
    ) -> PilotEvidenceCompletenessResult:
        if not protocol_hash:
            raise ValueError("PilotEvidenceCompletenessGate requires an explicit protocol_hash")
        self.evidence_store.create_run(run_id)
        evidence = self.evidence_store.load_run(run_id)
        records = [
            item
            for item in evidence.metric_records
            if item.run_id == run_id
            and (item.origin_run_id or item.run_id) == run_id
            and item.candidate_id == candidate_id
            and item.node_id == node_id
            and item.protocol_hash == protocol_hash
            and item.evidence_role == "current_observation"
            and item.inheritance_depth == 0
            and item.verified
        ]
        metric_names = {item.metric_name for item in records}
        missing_metrics = [name for name in self.required_metrics if name not in metric_names]
        missing_metrics.extend(
            prefix.rstrip("/")
            for prefix in self.required_metric_prefixes
            if not any(name.startswith(prefix) for name in metric_names)
        )

        node_artifacts = {
            entry.name: entry.path
            for entry in evidence.artifact_manifest
            if entry.run_id == run_id
            and entry.candidate_id == candidate_id
            and entry.node_id == node_id
            and entry.protocol_hash == protocol_hash
            and entry.name.startswith(f"{node_id}_")
            and entry.verify()
        }
        missing_artifacts = [
            suffix for suffix in self.required_artifacts if not any(name.endswith(suffix) for name in node_artifacts)
        ]
        report_path = next(
            (path for name, path in node_artifacts.items() if name.endswith("coco_error_report")),
            None,
        )
        report = _read_mapping(report_path)
        missing_report_fields = [name for name in self.required_report_fields if name not in report]
        contract = validate_coco_evidence_artifacts(
            predictions_path=next(
                (path for name, path in node_artifacts.items() if name.endswith("coco_predictions")),
                None,
            ),
            eval_path=next(
                (path for name, path in node_artifacts.items() if name.endswith("coco_eval")),
                None,
            ),
            error_report_path=report_path,
        )

        facts = [
            fact
            for fact in ErrorFactStore(self.evidence_store.root).read(run_id)
            if fact.run_id == run_id
            and fact.candidate_id == candidate_id
            and fact.node_id == node_id
            and fact.protocol_hash == protocol_hash
            and fact.evidence_role == "current_observation"
        ]
        fact_groups = {fact.fact_type for fact in facts}
        missing_fact_groups = [name for name in self.required_fact_groups if name not in fact_groups]
        actions: list[str] = []
        if "coco_predictions" in missing_artifacts:
            actions.append("run_coco_post_eval")
        if "coco_eval" in missing_artifacts or missing_metrics:
            actions.append("import_coco_eval")
        if "coco_error_report" in missing_artifacts or missing_report_fields:
            actions.append("mine_coco_errors")
        if contract.invalid_artifacts:
            actions.append("repair_coco_evidence_artifacts")
        if missing_fact_groups:
            actions.append("import_current_node_error_facts")

        return PilotEvidenceCompletenessResult(
            run_id=run_id,
            candidate_id=candidate_id,
            node_id=node_id,
            protocol_hash=protocol_hash,
            complete=not (
                missing_metrics
                or missing_artifacts
                or contract.invalid_artifacts
                or missing_report_fields
                or missing_fact_groups
            ),
            missing_metrics=missing_metrics,
            missing_artifacts=missing_artifacts,
            invalid_artifacts=contract.invalid_artifacts,
            artifact_hashes=contract.artifact_hashes,
            artifact_contract_hash=contract.contract_hash,
            missing_error_report_fields=missing_report_fields,
            missing_fact_groups=missing_fact_groups,
            evidence_actions=list(dict.fromkeys(actions)),
        )


class CocoEvidenceArtifactContractResult(BaseModel):
    """Content-addressed validation result for one node's COCO evidence bundle."""

    schema_version: str = "coco_evidence_artifact_contract.v1"
    valid: bool = False
    invalid_artifacts: dict[str, list[str]] = Field(default_factory=dict)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    contract_hash: str | None = None


def validate_coco_evidence_artifacts(
    *,
    predictions_path: Path | None,
    eval_path: Path | None,
    error_report_path: Path | None,
) -> CocoEvidenceArtifactContractResult:
    """Validate the semantic artifact contract, accepting an empty prediction list."""
    invalid: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    paths = {
        "predictions.json": predictions_path,
        "coco_eval.json": eval_path,
        "coco_error_report.json": error_report_path,
    }
    payloads: dict[str, Any] = {}
    for name, path in paths.items():
        if path is None or not path.is_file():
            continue
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            payloads[name] = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid[name] = [f"invalid_json:{exc.__class__.__name__}"]

    predictions = payloads.get("predictions.json")
    if predictions is not None:
        items = predictions.get("predictions") if isinstance(predictions, dict) else predictions
        errors: list[str] = []
        if not isinstance(items, list):
            errors.append("predictions_must_be_list")
        else:
            required = {"image_id", "category_id", "bbox", "score"}
            for index, item in enumerate(items):
                if not isinstance(item, dict) or not required.issubset(item):
                    errors.append(f"prediction_{index}_missing_required_fields")
                    break
        if errors:
            invalid["predictions.json"] = errors

    evaluation = payloads.get("coco_eval.json")
    if evaluation is not None:
        errors = []
        if not isinstance(evaluation, dict):
            errors.append("coco_eval_must_be_mapping")
        else:
            stats = evaluation.get("stats")
            for key, stats_index in (("AP_small", 3), ("AP_medium", 4), ("AP_large", 5)):
                if key not in evaluation and not (isinstance(stats, list) and len(stats) > stats_index):
                    errors.append(f"missing_{key}")
            for key in ("per_class_ap", "per_class_ar"):
                if not isinstance(evaluation.get(key), dict) or not evaluation[key]:
                    errors.append(f"missing_{key}")
        if errors:
            invalid["coco_eval.json"] = errors

    error_report = payloads.get("coco_error_report.json")
    if error_report is not None:
        errors = []
        if not isinstance(error_report, dict):
            errors.append("error_report_must_be_mapping")
        else:
            expected_types = {
                "false_negative_top_classes": list,
                "localization_error_top_classes": list,
                "background_false_positive_top_classes": list,
                "class_confusion_pairs": dict,
            }
            for key, expected_type in expected_types.items():
                if not isinstance(error_report.get(key), expected_type):
                    errors.append(f"invalid_{key}")
        if errors:
            invalid["coco_error_report.json"] = errors

    contract_hash = None
    if len(hashes) == len(paths):
        encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        contract_hash = hashlib.sha256(encoded).hexdigest()
    return CocoEvidenceArtifactContractResult(
        valid=len(hashes) == len(paths) and not invalid,
        invalid_artifacts=invalid,
        artifact_hashes=hashes,
        contract_hash=contract_hash,
    )


def _read_mapping(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
