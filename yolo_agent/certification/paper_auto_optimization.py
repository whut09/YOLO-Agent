"""Opt-in end-to-end acceptance for paper-driven automatic optimization."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterator, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from yolo_agent.agents.paper_outcome_learner import PaperOutcomeLearningResult
from yolo_agent.certification.paper_auto_optimization_evidence import (
    PaperPilotEvidenceBundle,
    validate_paper_pilot_evidence,
)
from yolo_agent.certification.paper_auto_optimization_maturity import (
    promote_sampling_pilot_reproduced,
)
from yolo_agent.certification.paper_auto_optimization_memory import (
    record_sampling_pilot_outcome,
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
    _asha_observation,
    _certification_scheduler,
    _import_bootstrap_metrics,
    _import_observation,
    _run_paired_bootstrap,
    _target_error_facts,
    _validate_runtime_artifacts,
    _node,
)
from yolo_agent.certification.code_identity import certification_code_hash
from yolo_agent.certification.fixture import create_mini_coco_fixture
from yolo_agent.certification.schemas import CertificationPromotionResult
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.paired_bootstrap import PairedBootstrapReport
from yolo_agent.core.paired_experiment import (
    PairedExperimentResult,
    build_paired_experiment_result,
)
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload


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

    def run(
        self,
        *,
        workdir: Path | str,
        research_root: Path | str = "research",
        source: Path | str | None = None,
        maturity_registry: Path | str = "runs/component_maturity_registry.yaml",
        policy_memory_root: Path | str = "runs",
        model: str = "yolo26n.pt",
        device: str = "0",
        source_commit: str | None = None,
        execute_real_gpu: bool = False,
    ) -> PaperAutoOptimizationReport:
        """Run one sampling pilot_3 -> ASHA -> pilot_10 acceptance chain."""
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with _acceptance_workdir_lock(root):
            return self._run_locked(
                root=root,
                research_root=research_root,
                source=source,
                maturity_registry=maturity_registry,
                policy_memory_root=policy_memory_root,
                model=model,
                device=device,
                source_commit=source_commit,
                execute_real_gpu=execute_real_gpu,
            )

    def _run_locked(
        self,
        *,
        root: Path,
        research_root: Path | str,
        source: Path | str | None,
        maturity_registry: Path | str,
        policy_memory_root: Path | str,
        model: str,
        device: str,
        source_commit: str | None,
        execute_real_gpu: bool,
    ) -> PaperAutoOptimizationReport:
        if not execute_real_gpu:
            return self._write_report(
                root,
                PaperAutoOptimizationReport(
                    acceptance_id=root.name or "paper-auto-optimization",
                    status="skipped",
                    execute_real_gpu=False,
                    model=model,
                    device=device,
                    stages=[
                        _stage(
                            "certified_adapter",
                            status="skipped",
                            message="Pass --execute-real-gpu to opt in.",
                        )
                    ],
                    failures=["real_gpu_execution_not_confirmed"],
                ),
            )

        stages: list[PaperAutoOptimizationStage] = []
        pairs: list[PaperPilotPairResult] = []
        research: PaperAcceptanceResearchContext | None = None
        protocol_hash: str | None = None
        objective_hash = hash_payload(
            {
                "primary_metric": "ap_small",
                "target_metrics": ["per_class_ar/object"],
                "target_error_facts": ["false_negative/object"],
            }
        )
        try:
            data_yaml = create_mini_coco_fixture(root / "mini_coco")
            research_path = root / "research_context.yaml"
            research = self._resolve_preparer(
                research_root=research_root,
                source=source,
                maturity_registry=maturity_registry,
                source_commit=source_commit,
            ).prepare(research_path)
            stages.append(
                _stage(
                    "fresh_snapshot",
                    artifacts={
                        "research_context": research_path.as_posix(),
                        "snapshot": (
                            research.snapshot_path / "snapshot.yaml"
                        ).as_posix(),
                    },
                    metrics={
                        "snapshot_hash": research.snapshot_hash,
                        "source_commit": research.source_commit,
                    },
                )
            )
            stages.append(
                _stage(
                    "diagnosis",
                    message="Acceptance objective targets small-object false negatives.",
                    metrics={
                        "symptom": "small_object_false_negative",
                        "target_metrics": ["ap_small", "per_class_ar/object"],
                        "target_error_facts": ["false_negative/object"],
                        "paper_claim_used_as_local_evidence": False,
                    },
                )
            )
            stages.append(
                _stage(
                    "method_profile",
                    metrics={
                        "paper_ids": research.paper_ids,
                        "profile_ids": research.method_profile_ids,
                        "component_id": research.component_id,
                        "adaptation": "component_adaptation",
                    },
                )
            )
            stages.append(
                _stage(
                    "certified_adapter",
                    metrics={
                        "component_id": research.component_id,
                        "adapter_hash": research.adapter_hash,
                        "maturity": research.maturity,
                        "maturity_protocol_hash": research.maturity_protocol_hash,
                        "scalar_hpo_enabled": False,
                    },
                )
            )
            environment = self.backend.environment()
            if not bool(environment.get("cuda_available")):
                raise RuntimeError("CUDA is unavailable for paper acceptance")
            protocol_hash = hash_payload(
                {
                    "suite": "paper_auto_optimization_acceptance.v1",
                    "snapshot_hash": research.snapshot_hash,
                    "adapter_hash": research.adapter_hash,
                    "objective_hash": objective_hash,
                    "model": model,
                    "imgsz": 640,
                    "code_hash": certification_code_hash(),
                }
            )
            store = EvidenceStore(root / "evidence")
            run_id = root.name or "paper_auto_optimization"
            pilot_3 = self._run_pilot_pair(
                root=root,
                store=store,
                run_id=run_id,
                stage_id="pilot_3",
                epochs=3,
                seed=1,
                base_protocol_hash=protocol_hash,
                objective_hash=objective_hash,
                environment=environment,
                data_yaml=data_yaml,
                model=model,
                device=device,
            )
            pairs.append(pilot_3)
            stages.extend(_pilot_3_stages(pilot_3))

            scheduler = _certification_scheduler(
                run_id,
                cohort_size=1,
                target_error_required=True,
            )
            scheduler.register_trial(
                trial_id="sampling.small_object",
                candidate_id="sampling.small_object",
                source_run_id=run_id,
                source_node=_node(
                    "sampling.small_object",
                    pilot_3.candidate_run.node_id,
                    dict(SAMPLING_ACCEPTANCE_RECIPE.overrides),
                ),
                baseline_control_node=_node(
                    pilot_3.control_run.candidate_id,
                    pilot_3.control_run.node_id,
                    {},
                ),
                target_error_facts=_target_error_facts(
                    SAMPLING_ACCEPTANCE_RECIPE
                ),
            )
            scheduler.report(
                "sampling.small_object",
                _asha_observation(
                    "pilot_3",
                    pilot_3.paired,
                    seed=1,
                    primary_metric="ap_small",
                    promotion=pilot_3.promotion,
                ),
            )
            assignment = scheduler.next_assignment()
            if assignment is None or assignment.stage_id != "pilot_10":
                reason = ", ".join(pilot_3.promotion.rejection_reasons)
                failure_reason = (
                    "ASHA eliminated sampling.small_object after pilot_3"
                    + (f": {reason}" if reason else "")
                )
                learning, memory_path = _record_policy_outcome(
                    root=root,
                    policy_memory_root=policy_memory_root,
                    run_id=run_id,
                    research=research,
                    protocol=pilot_3.protocol,
                    pilot_3=pilot_3,
                    pilot_10=None,
                    failure_reason=failure_reason,
                )
                stages.append(_policy_memory_stage(root, memory_path, learning))
                raise RuntimeError(failure_reason)
            stages.append(
                _stage(
                    "asha",
                    metrics={
                        "assignment_id": assignment.assignment_id,
                        "survivor": assignment.candidate_id,
                        "budget_authority": "ASHA",
                        "scalar_hpo_enabled": False,
                    },
                )
            )

            pilot_10 = self._run_pilot_pair(
                root=root,
                store=store,
                run_id=run_id,
                stage_id="pilot_10",
                epochs=10,
                seed=1,
                base_protocol_hash=protocol_hash,
                objective_hash=objective_hash,
                environment=environment,
                data_yaml=data_yaml,
                model=model,
                device=device,
            )
            pairs.append(pilot_10)
            failure_reason = None
            if not pilot_10.promotion.passed:
                failure_reason = (
                    "pilot_10 promotion rejected: "
                    + ", ".join(pilot_10.promotion.rejection_reasons)
                )
            learning, memory_path = _record_policy_outcome(
                root=root,
                policy_memory_root=policy_memory_root,
                run_id=run_id,
                research=research,
                protocol=pilot_10.protocol,
                pilot_3=pilot_3,
                pilot_10=pilot_10,
                failure_reason=failure_reason,
            )
            stages.append(_policy_memory_stage(root, memory_path, learning))
            if failure_reason is not None:
                raise RuntimeError(failure_reason)
            trial = scheduler.report(
                "sampling.small_object",
                _asha_observation(
                    "pilot_10",
                    pilot_10.paired,
                    seed=1,
                    primary_metric="ap_small",
                    promotion=pilot_10.promotion,
                ),
            )
            if trial.status != "full_pending_confirmation":
                raise RuntimeError(
                    "ASHA did not stop at explicit full-run consent boundary"
                )
            stages.append(_pilot_10_stage(pilot_10))

            maturity_artifact = root / "artifacts" / "pilot_reproduced.yaml"
            promoted = promote_sampling_pilot_reproduced(
                registry_path=maturity_registry,
                research=research,
                acceptance_protocol_hash=protocol_hash,
                pilot_3=pilot_3.summary,
                pilot_10=pilot_10.summary,
                output_path=maturity_artifact,
            )
            stages.append(
                _stage(
                    "pilot_reproduced",
                    message=(
                        "Local paired pilots passed; full and multi-seed remain "
                        "behind --confirm-full-run."
                    ),
                    artifacts={"maturity_evidence": maturity_artifact.as_posix()},
                    metrics={
                        "component_id": promoted.component_id,
                        "evidence_hash": promoted.evidence_hash,
                        "next_boundary": "explicit_full_run_consent",
                    },
                )
            )
            payload = AdapterRuntimePayload.read(
                pilot_10.candidate_run.runtime_artifacts["runtime_payload"]
            )
            return self._write_report(
                root,
                PaperAutoOptimizationReport(
                    acceptance_id=run_id,
                    status="passed",
                    execute_real_gpu=True,
                    model=model,
                    device=device,
                    research_snapshot_hash=research.snapshot_hash,
                    research_snapshot_path=research.snapshot_path,
                    paper_ids=research.paper_ids,
                    adapter_hash=research.adapter_hash,
                    maturity="pilot_reproduced",
                    runtime_payload_hash=payload.payload_hash,
                    objective_hash=objective_hash,
                    protocol_hash=protocol_hash,
                    stages=stages,
                    protocol_identities={
                        item.stage_id: item.protocol for item in pairs
                    },
                    paired_deltas=[item.summary for item in pairs],
                    asha_survivor="sampling.small_object",
                    policy_memory_path=memory_path,
                    pilot_reproduced=True,
                ),
            )
        except PaperEvidenceRecoveryRequired as exc:
            stages.append(
                _stage(
                    "evidence_recovery",
                    status="recovery",
                    message=str(exc),
                    metrics={"actions": exc.actions, "training_allowed": False},
                )
            )
            return self._write_report(
                root,
                _partial_report(
                    root=root,
                    status="recovery",
                    model=model,
                    device=device,
                    research=research,
                    protocol_hash=protocol_hash,
                    objective_hash=objective_hash,
                    stages=stages,
                    pairs=pairs,
                    failures=[],
                    recovery_actions=exc.actions,
                ),
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._write_report(
                root,
                _partial_report(
                    root=root,
                    status="failed",
                    model=model,
                    device=device,
                    research=research,
                    protocol_hash=protocol_hash,
                    objective_hash=objective_hash,
                    stages=stages,
                    pairs=pairs,
                    failures=[str(exc)],
                ),
            )

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


@contextmanager
def _acceptance_workdir_lock(root: Path) -> Iterator[None]:
    """Prevent concurrent acceptance processes from contaminating one workdir."""
    lock_path = root / ".paper_auto_optimization.lock"
    token = uuid4().hex
    payload = json.dumps({"pid": os.getpid(), "token": token})
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        owner = lock_path.read_text(encoding="utf-8-sig", errors="replace")
        raise RuntimeError(
            "paper auto-optimization workdir is already active; "
            f"choose a fresh --workdir or wait for its owner: {owner}"
        ) from exc
    try:
        os.write(descriptor, payload.encode("utf-8"))
        os.close(descriptor)
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            current = {}
        if current.get("token") == token:
            lock_path.unlink(missing_ok=True)


def _record_policy_outcome(
    *,
    root: Path,
    policy_memory_root: Path | str,
    run_id: str,
    research: PaperAcceptanceResearchContext,
    protocol: PaperProtocolIdentity,
    pilot_3: PaperPilotPairResult,
    pilot_10: PaperPilotPairResult | None,
    failure_reason: str | None,
) -> tuple[PaperOutcomeLearningResult, Path]:
    memory_artifact = root / "artifacts" / "policy_memory_update.json"
    learning = record_sampling_pilot_outcome(
        memory_root=policy_memory_root,
        run_id=run_id,
        research=research,
        protocol=protocol,
        pilot_3=pilot_3.paired,
        pilot_10=pilot_10.paired if pilot_10 is not None else None,
        output_path=memory_artifact,
        failure_reason=failure_reason,
    )
    return learning, Path(policy_memory_root) / "policy_memory.jsonl"


def _policy_memory_stage(
    root: Path,
    memory_path: Path,
    learning: PaperOutcomeLearningResult,
) -> PaperAutoOptimizationStage:
    return _stage(
        "policy_memory",
        artifacts={
            "learning_result": (
                root / "artifacts" / "policy_memory_update.json"
            ).as_posix(),
            "policy_memory": memory_path.as_posix(),
        },
        metrics={
            "record_id": learning.record.record_id,
            "local_posterior_status": learning.local_posterior_status,
            "failure_reason": learning.record.failure_reason,
            "paper_prior_is_local_evidence": False,
        },
    )


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


def _pilot_3_stages(pair: PaperPilotPairResult) -> list[PaperAutoOptimizationStage]:
    return [
        _stage(
            "matched_pilot_3_cohort",
            artifacts={
                "control_checkpoint": pair.control_run.checkpoint.as_posix(),
                "candidate_checkpoint": pair.candidate_run.checkpoint.as_posix(),
            },
            metrics={
                "cohort_size": 1,
                "control": pair.control_run.candidate_id,
                "candidate": pair.candidate_run.candidate_id,
                "protocol_hash": pair.protocol.protocol_hash,
                "protocol_matched": True,
            },
        ),
        _stage(
            "candidate_control_post_eval",
            artifacts={
                "control_predictions": pair.control_evaluation.predictions_path.as_posix(),
                "control_coco_eval": pair.control_evaluation.eval_path.as_posix(),
                "candidate_predictions": pair.candidate_evaluation.predictions_path.as_posix(),
                "candidate_coco_eval": pair.candidate_evaluation.eval_path.as_posix(),
            },
            metrics={"stage": "pilot_3", "evaluated_nodes": 2},
        ),
        _stage(
            "complete_coco_error_facts",
            artifacts={
                "evidence_contract": (
                    Path(pair.candidate_evaluation.eval_path).parents[2]
                    / "evidence_contracts"
                    / "pilot_3.yaml"
                ).as_posix()
            },
            metrics={
                "candidate_fact_count": pair.evidence.candidate_fact_count,
                "baseline_fact_count": pair.evidence.baseline_fact_count,
                "current_run_only": True,
                "current_node_only": True,
                "same_protocol_hash": True,
            },
        ),
        _stage(
            "paired_bootstrap_delta",
            metrics=pair.summary.model_dump(mode="json"),
        ),
    ]


def _pilot_10_stage(pair: PaperPilotPairResult) -> PaperAutoOptimizationStage:
    return _stage(
        "pilot_10",
        artifacts={
            "control_checkpoint": pair.control_run.checkpoint.as_posix(),
            "candidate_checkpoint": pair.candidate_run.checkpoint.as_posix(),
        },
        metrics={
            **pair.summary.model_dump(mode="json"),
            "protocol_hash": pair.protocol.protocol_hash,
            "protocol_matched": True,
            "full_training_started": False,
        },
    )


def _partial_report(
    *,
    root: Path,
    status: PaperAutoOptimizationStatus,
    model: str,
    device: str,
    research: PaperAcceptanceResearchContext | None,
    protocol_hash: str | None,
    objective_hash: str,
    stages: list[PaperAutoOptimizationStage],
    pairs: list[PaperPilotPairResult],
    failures: list[str],
    recovery_actions: list[str] | None = None,
) -> PaperAutoOptimizationReport:
    runtime_hash = None
    if pairs:
        payload_path = pairs[-1].candidate_run.runtime_artifacts.get("runtime_payload")
        if payload_path is not None and payload_path.is_file():
            runtime_hash = AdapterRuntimePayload.read(payload_path).payload_hash
    return PaperAutoOptimizationReport(
        acceptance_id=root.name or "paper-auto-optimization",
        status=status,
        execute_real_gpu=True,
        model=model,
        device=device,
        research_snapshot_hash=(research.snapshot_hash if research else None),
        research_snapshot_path=(research.snapshot_path if research else None),
        paper_ids=(research.paper_ids if research else []),
        adapter_hash=(research.adapter_hash if research else None),
        maturity=(research.maturity if research else None),
        runtime_payload_hash=runtime_hash,
        objective_hash=objective_hash,
        protocol_hash=protocol_hash,
        stages=stages,
        protocol_identities={item.stage_id: item.protocol for item in pairs},
        paired_deltas=[item.summary for item in pairs],
        failures=failures,
        evidence_recovery_actions=recovery_actions or [],
    )


__all__ = [
    "PaperAutoOptimizationAcceptanceSuite",
    "PaperResearchPreparerProtocol",
]
