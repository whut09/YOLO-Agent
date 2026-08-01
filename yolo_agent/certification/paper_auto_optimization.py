"""Opt-in end-to-end acceptance for paper-driven automatic optimization."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from yolo_agent.certification.paper_auto_optimization_evidence import (
    PaperPilotEvidenceBundle,
    validate_paper_pilot_evidence,
)
from yolo_agent.certification.paper_auto_optimization_promotion import (
    SAMPLING_ACCEPTANCE_RECIPE,
    evaluate_sampling_promotion,
)
from yolo_agent.certification.paper_auto_optimization_protocol import (
    build_paper_protocol_identity,
    compare_paper_protocols,
    hash_payload,
)
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    PaperAcceptanceResearchPreparer,
)
from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
    PaperAutoOptimizationStage,
    PaperAutoOptimizationStageId,
    PaperAutoOptimizationStatus,
    PaperPairedDelta,
    PaperProtocolIdentity,
)
from yolo_agent.certification.runner import (
    BackendEvaluation,
    BackendRun,
    GpuAcceptanceBackend,
    UltralyticsGpuBackend,
    _additional_metrics,
    _import_bootstrap_metrics,
    _import_observation,
    _run_paired_bootstrap,
    _target_error_facts,
    _validate_runtime_artifacts,
)
from yolo_agent.certification.schemas import CertificationPromotionResult
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.paired_bootstrap import PairedBootstrapReport
from yolo_agent.core.paired_experiment import (
    PairedExperimentResult,
    build_paired_experiment_result,
)


class PaperResearchPreparerProtocol(Protocol):
    def prepare(self, output_path: Path | str) -> PaperAcceptanceResearchContext: ...


class PaperEvidenceRecoveryRequired(RuntimeError):
    """Stop training and surface queue-owned evidence recovery actions."""

    def __init__(self, message: str, actions: list[str]) -> None:
        super().__init__(message)
        self.actions = list(dict.fromkeys(actions or ["recover_coco_post_eval"]))


class PaperPilotPairResult(BaseModel):
    """Internal complete result for one candidate/control fidelity."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage_id: str
    protocol: PaperProtocolIdentity
    control_run: BackendRun
    candidate_run: BackendRun
    control_evaluation: BackendEvaluation
    candidate_evaluation: BackendEvaluation
    evidence: PaperPilotEvidenceBundle
    bootstrap: PairedBootstrapReport
    paired: PairedExperimentResult
    promotion: CertificationPromotionResult
    summary: PaperPairedDelta


class PaperAutoOptimizationAcceptanceSuite:
    """Certify one real paper recipe without granting full-run consent."""

    report_name = "paper_auto_optimization_report.yaml"

    def __init__(
        self,
        backend: GpuAcceptanceBackend | None = None,
        research_preparer: PaperResearchPreparerProtocol | None = None,
    ) -> None:
        self.backend = backend or UltralyticsGpuBackend()
        self.research_preparer = research_preparer

    def _run_pilot_pair(
        self,
        *,
        root: Path,
        store: EvidenceStore,
        run_id: str,
        stage_id: str,
        epochs: int,
        seed: int,
        base_protocol_hash: str,
        objective_hash: str,
        environment: dict[str, Any],
        data_yaml: Path,
        model: str,
        device: str,
    ) -> PaperPilotPairResult:
        protocol_hash = hash_payload(
            {"base_protocol_hash": base_protocol_hash, "fidelity": stage_id}
        )
        expected_protocol = build_paper_protocol_identity(
            data_yaml=data_yaml,
            protocol_hash=protocol_hash,
            objective_hash=objective_hash,
            epochs=epochs,
            seed=seed,
            ultralytics_version=str(environment["ultralytics_version"]),
        )
        control = self.backend.train(
            candidate_id=f"baseline_{stage_id}",
            node_id=f"baseline_{stage_id}",
            data_yaml=data_yaml,
            model=model,
            workdir=root,
            device=device,
            epochs=epochs,
            seed=seed,
            protocol_hash=protocol_hash,
            overrides={},
        )
        candidate = self.backend.train(
            candidate_id="sampling.small_object",
            node_id=f"sampling_small_object_{stage_id}",
            data_yaml=data_yaml,
            model=model,
            workdir=root,
            device=device,
            epochs=epochs,
            seed=seed,
            protocol_hash=protocol_hash,
            overrides=dict(SAMPLING_ACCEPTANCE_RECIPE.overrides),
        )
        protocol_match = compare_paper_protocols(
            control.protocol_identity,
            candidate.protocol_identity,
        )
        if not protocol_match.matched:
            raise RuntimeError(
                "candidate/control protocol mismatch: "
                + ", ".join(sorted(protocol_match.mismatched_fields))
            )
        if control.protocol_identity != expected_protocol:
            raise RuntimeError("backend protocol identity differs from acceptance protocol")
        runtime_artifacts, _ = _validate_runtime_artifacts(
            candidate,
            protocol_hash=protocol_hash,
        )
        if not runtime_artifacts:
            raise RuntimeError("sampling runtime artifacts are empty")

        try:
            control_eval = self.backend.evaluate(
                run=control,
                data_yaml=data_yaml,
                workdir=root,
                device=device,
            )
            candidate_eval = self.backend.evaluate(
                run=candidate,
                data_yaml=data_yaml,
                workdir=root,
                device=device,
            )
            identity = _evidence_identity(expected_protocol, stage_id)
            _import_observation(
                store,
                run_id,
                control,
                control_eval,
                identity,
                "baseline_reference",
            )
            _import_observation(
                store,
                run_id,
                candidate,
                candidate_eval,
                identity,
                "current_observation",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise PaperEvidenceRecoveryRequired(
                f"{stage_id} COCO post-eval incomplete: {exc}",
                [
                    "recover_control_coco_post_eval",
                    "recover_candidate_coco_post_eval",
                ],
            ) from exc

        evidence = validate_paper_pilot_evidence(
            store=store,
            run_id=run_id,
            stage_id=stage_id,
            protocol_hash=protocol_hash,
            control_run=control,
            control_evaluation=control_eval,
            candidate_run=candidate,
            candidate_evaluation=candidate_eval,
            output_path=root / "evidence_contracts" / f"{stage_id}.yaml",
        )
        if not evidence.complete:
            raise PaperEvidenceRecoveryRequired(
                f"{stage_id} evidence is incomplete: "
                + ", ".join(evidence.missing_evidence),
                evidence.recovery_actions,
            )

        bootstrap_path = root / "paired_bootstrap" / f"{stage_id}.json"
        bootstrap = _run_paired_bootstrap(
            data_yaml=data_yaml,
            control=control_eval,
            candidate=candidate_eval,
            output=bootstrap_path,
            seed=seed + epochs,
        )
        if bootstrap.status != "completed":
            raise PaperEvidenceRecoveryRequired(
                f"{stage_id} paired bootstrap is incomplete",
                ["recover_paired_bootstrap"],
            )
        _import_bootstrap_metrics(
            store,
            run_id=run_id,
            run=candidate,
            identity=_evidence_identity(expected_protocol, stage_id),
            report=bootstrap,
        )
        evidence_set = store.load_run(run_id)
        paired = build_paired_experiment_result(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            candidate_node_id=candidate.node_id,
            metric_records=evidence_set.metric_records,
            error_facts=ErrorFactStore(store.root).read(run_id),
            primary_metric=SAMPLING_ACCEPTANCE_RECIPE.primary_metric,
            target_error_facts=_target_error_facts(SAMPLING_ACCEPTANCE_RECIPE),
            additional_metrics=_additional_metrics(SAMPLING_ACCEPTANCE_RECIPE),
        )
        if not paired.verified:
            raise PaperEvidenceRecoveryRequired(
                f"{stage_id} paired result incomplete: " + ", ".join(paired.blockers),
                ["recover_verified_paired_delta"],
            )
        promotion, summary = evaluate_sampling_promotion(
            stage_id=stage_id,
            paired=paired,
            control=control_eval,
            candidate=candidate_eval,
            bootstrap=bootstrap,
            paired_result_path=root / "paired_results" / f"{stage_id}.json",
        )
        return PaperPilotPairResult(
            stage_id=stage_id,
            protocol=expected_protocol,
            control_run=control,
            candidate_run=candidate,
            control_evaluation=control_eval,
            candidate_evaluation=candidate_eval,
            evidence=evidence,
            bootstrap=bootstrap,
            paired=paired,
            promotion=promotion,
            summary=summary,
        )

    def _resolve_preparer(
        self,
        *,
        research_root: Path | str,
        source: Path | str | None,
        maturity_registry: Path | str,
        source_commit: str | None,
    ) -> PaperResearchPreparerProtocol:
        if self.research_preparer is not None:
            return self.research_preparer
        return PaperAcceptanceResearchPreparer(
            research_root=research_root,
            source=source,
            maturity_registry=maturity_registry,
            source_commit=source_commit,
        )

    @classmethod
    def _write_report(
        cls,
        root: Path,
        report: PaperAutoOptimizationReport,
    ) -> PaperAutoOptimizationReport:
        path = root / cls.report_name
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        report.to_yaml(temporary, exclude_none=True, sort_keys=False)
        temporary.replace(path)
        return report


def _stage(
    stage_id: PaperAutoOptimizationStageId,
    *,
    status: PaperAutoOptimizationStatus = "passed",
    message: str = "",
    command: list[str] | None = None,
    artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> PaperAutoOptimizationStage:
    now = datetime.now(timezone.utc)
    return PaperAutoOptimizationStage(
        stage_id=stage_id,
        status=status,
        message=message,
        command=command or [],
        artifacts=artifacts or {},
        metrics=metrics or {},
        started_at=now,
        completed_at=now,
    )


def _evidence_identity(
    protocol: PaperProtocolIdentity,
    fidelity: str,
) -> dict[str, Any]:
    return {
        "protocol_hash": protocol.protocol_hash,
        "dataset_manifest_sha256": protocol.dataset_manifest_hash,
        "subset_manifest_sha256": protocol.subset_manifest_hash,
        "eval_protocol_hash": protocol.eval_protocol_hash,
        "seed": protocol.seed,
        "fidelity": fidelity,
        "epochs": protocol.epochs,
        "batch_policy_hash": protocol.batch_policy_hash,
        "ultralytics_version": protocol.ultralytics_version,
        "imgsz": protocol.imgsz,
    }


__all__ = [
    "PaperAutoOptimizationAcceptanceSuite",
    "PaperResearchPreparerProtocol",
]
