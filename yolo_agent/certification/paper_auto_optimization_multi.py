"""Four-family paper auto-optimization acceptance state machine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from yolo_agent.agents.paper_outcome_learner import PaperOutcomeLearningResult
from yolo_agent.certification.component_runtime_evidence import (
    validate_component_runtime_artifacts,
)
from yolo_agent.certification.fixture import create_mini_coco_fixture
from yolo_agent.certification.paper_auto_optimization_evidence import (
    PaperPilotEvidenceBundle,
    validate_paper_pilot_evidence,
)
from yolo_agent.certification.paper_auto_optimization_maturity import (
    promote_component_pilot_reproduced,
)
from yolo_agent.certification.paper_auto_optimization_memory import (
    record_paper_pilot_outcome,
)
from yolo_agent.certification.paper_auto_optimization_promotion import (
    evaluate_paper_recipe_promotion,
)
from yolo_agent.certification.paper_auto_optimization_protocol import (
    build_paper_protocol_identity,
    compare_paper_protocols,
    hash_payload,
)
from yolo_agent.certification.paper_auto_optimization_research import (
    PaperAcceptanceResearchContext,
    PaperAcceptanceTrackContext,
)
from yolo_agent.certification.paper_auto_optimization_schemas import (
    PaperAutoOptimizationReport,
    PaperAutoOptimizationStage,
    PaperAutoOptimizationStageId,
    PaperPairedDelta,
    PaperProtocolIdentity,
)
from yolo_agent.certification.paper_auto_optimization_tracks import (
    PAPER_ACCEPTANCE_RECIPES,
    PaperAcceptanceRecipe,
)
from yolo_agent.certification.runner import (
    BackendEvaluation,
    BackendRun,
    GpuAcceptanceBackend,
    _asha_observation,
    _certification_scheduler,
    _import_bootstrap_metrics,
    _import_observation,
    _node,
    _run_paired_bootstrap,
)
from yolo_agent.certification.schemas import CertificationPromotionResult
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.core.error_facts import ErrorFactStore
from yolo_agent.core.evidence_store import EvidenceStore
from yolo_agent.core.paired_bootstrap import PairedBootstrapReport
from yolo_agent.core.paired_experiment import (
    PairedExperimentResult,
    build_paired_experiment_result,
)


class PaperResearchPreparer(Protocol):
    def prepare(self, output_path: Path | str) -> PaperAcceptanceResearchContext: ...


class PaperEvidenceRecoveryRequired(RuntimeError):
    def __init__(self, message: str, actions: list[str]) -> None:
        super().__init__(message)
        self.actions = list(dict.fromkeys(actions or ["recover_coco_post_eval"]))


class MultiMechanismPilotPair(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    recipe: PaperAcceptanceRecipe
    track: PaperAcceptanceTrackContext
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


def run_multi_mechanism_acceptance(
    *,
    root: Path,
    backend: GpuAcceptanceBackend,
    research_preparer: PaperResearchPreparer | None,
    maturity_registry: Path | str,
    policy_memory_root: Path | str,
    model: str,
    device: str,
    execute_real_gpu: bool,
) -> PaperAutoOptimizationReport:
    """Run the bounded four-family cohort without requesting full-run consent."""
    acceptance_id = root.name or "paper-auto-optimization"
    if not execute_real_gpu:
        return PaperAutoOptimizationReport(
            acceptance_id=acceptance_id,
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
        )
    if research_preparer is None:
        raise RuntimeError("paper acceptance requires a research preparer")

    stages: list[PaperAutoOptimizationStage] = []
    pairs: list[MultiMechanismPilotPair] = []
    learning_results: list[PaperOutcomeLearningResult] = []
    research: PaperAcceptanceResearchContext | None = None
    acceptance_protocol_hash: str | None = None
    objective_hash = hash_payload(
        {
            "recipes": [
                {
                    "recipe_id": item.recipe_id,
                    "primary_metric": item.primary_metric,
                    "target_metrics": item.target_metrics,
                    "target_error_facts": item.target_error_facts,
                }
                for item in PAPER_ACCEPTANCE_RECIPES
            ]
        }
    )
    reproduced: list[str] = []
    maturity_artifacts: dict[str, str] = {}
    try:
        data_yaml = create_mini_coco_fixture(root / "mini_coco")
        research_path = root / "research_context.yaml"
        research = research_preparer.prepare(research_path)
        tracks = _validated_tracks(research)
        stages.extend(_research_stages(research, tracks, research_path))
        environment = backend.environment()
        if not bool(environment.get("cuda_available")):
            raise RuntimeError("CUDA is unavailable for paper acceptance")
        acceptance_protocol_hash = hash_payload(
            {
                "suite": "paper_auto_optimization_acceptance.v2",
                "snapshot_hash": research.snapshot_hash,
                "objective_hash": objective_hash,
                "components": {
                    item.component_id: {
                        "adapter_hash": item.adapter_hash,
                        "maturity_protocol_hash": item.maturity_protocol_hash,
                    }
                    for item in tracks.values()
                },
                "model": model,
                "imgsz": 640,
            }
        )
        run_id = acceptance_id
        store = EvidenceStore(root / "evidence")
        scheduler = _certification_scheduler(
            run_id,
            cohort_size=len(PAPER_ACCEPTANCE_RECIPES),
            target_error_required=True,
        )
        scheduler.study.confirmation_seeds = [1, 2, 3]
        for recipe in PAPER_ACCEPTANCE_RECIPES:
            scheduler.register_trial(
                trial_id=recipe.component_id,
                candidate_id=recipe.component_id,
                source_run_id=run_id,
                source_node=_node(
                    recipe.component_id,
                    f"{_slug(recipe.component_id)}_pilot_3",
                    {recipe.changed_variable: recipe.adapter_options},
                ),
                baseline_control_node=_node(
                    f"baseline_{recipe.component_id}",
                    f"baseline_{_slug(recipe.component_id)}_pilot_3",
                    {},
                ),
                target_error_facts=recipe.target_error_facts,
            )

        pilot_3_pairs: dict[str, MultiMechanismPilotPair] = {}
        for recipe in PAPER_ACCEPTANCE_RECIPES:
            pair = _run_pair(
                root=root,
                backend=backend,
                store=store,
                run_id=run_id,
                recipe=recipe,
                track=tracks[recipe.component_id],
                stage_id="pilot_3",
                epochs=3,
                seed=1,
                base_protocol_hash=acceptance_protocol_hash,
                objective_hash=objective_hash,
                environment=environment,
                data_yaml=data_yaml,
                model=model,
                device=device,
            )
            pairs.append(pair)
            pilot_3_pairs[recipe.component_id] = pair
            scheduler.report(
                recipe.component_id,
                _asha_observation(
                    "pilot_3",
                    pair.paired,
                    seed=1,
                    primary_metric=recipe.primary_metric,
                    promotion=pair.promotion,
                ),
            )
        stages.extend(_pilot_3_cohort_stages(root, list(pilot_3_pairs.values())))

        assignment = scheduler.next_assignment(confirm_full_run=False)
        promoted_ids = {
            assignment.candidate_id
            if assignment is not None and assignment.stage_id == "pilot_10"
            else ""
        }
        for trial in scheduler.study.trials:
            if (
                trial.status == "waiting"
                and trial.pending_stage is None
                and trial.candidate_id not in promoted_ids
            ):
                trial.status = "eliminated"
                trial.eliminated_reason = "asha_budget_pruned_after_pilot_3"
            if trial.status != "eliminated":
                continue
            pair = pilot_3_pairs[trial.candidate_id]
            learning_results.append(
                _record_outcome(
                    root=root,
                    policy_memory_root=policy_memory_root,
                    run_id=run_id,
                    research=research,
                    pair_3=pair,
                    pair_10=None,
                    failure_reason=trial.eliminated_reason or "eliminated_after_pilot_3",
                )
            )

        assignments = []
        while assignment is not None:
            if assignment is None or assignment.stage_id != "pilot_10":
                break
            assignments.append(assignment)
            recipe = next(
                item
                for item in PAPER_ACCEPTANCE_RECIPES
                if item.component_id == assignment.candidate_id
            )
            pair_10 = _run_pair(
                root=root,
                backend=backend,
                store=store,
                run_id=run_id,
                recipe=recipe,
                track=tracks[recipe.component_id],
                stage_id="pilot_10",
                epochs=10,
                seed=1,
                base_protocol_hash=acceptance_protocol_hash,
                objective_hash=objective_hash,
                environment=environment,
                data_yaml=data_yaml,
                model=model,
                device=device,
            )
            pairs.append(pair_10)
            trial = scheduler.report(
                recipe.component_id,
                _asha_observation(
                    "pilot_10",
                    pair_10.paired,
                    seed=1,
                    primary_metric=recipe.primary_metric,
                    promotion=pair_10.promotion,
                ),
            )
            failure_reason = (
                trial.eliminated_reason or "eliminated_after_pilot_10"
                if trial.status == "eliminated"
                else None
            )
            learning_results.append(
                _record_outcome(
                    root=root,
                    policy_memory_root=policy_memory_root,
                    run_id=run_id,
                    research=research,
                    pair_3=pilot_3_pairs[recipe.component_id],
                    pair_10=pair_10,
                    failure_reason=failure_reason,
                )
            )
            if trial.status != "full_pending_confirmation":
                assignment = scheduler.next_assignment(confirm_full_run=False)
                continue
            maturity_path = (
                root
                / "artifacts"
                / "pilot_reproduced"
                / f"{_slug(recipe.component_id)}.yaml"
            )
            promote_component_pilot_reproduced(
                registry_path=maturity_registry,
                research=research,
                track=tracks[recipe.component_id],
                recipe=recipe,
                acceptance_protocol_hash=acceptance_protocol_hash,
                pilot_3=pilot_3_pairs[recipe.component_id].summary,
                pilot_10=pair_10.summary,
                output_path=maturity_path,
            )
            reproduced.append(recipe.component_id)
            maturity_artifacts[recipe.component_id] = maturity_path.as_posix()
            assignment = scheduler.next_assignment(confirm_full_run=False)

        stages.append(
            _stage(
                "asha",
                status="passed" if assignments else "failed",
                metrics={
                    "budget_authority": "ASHA",
                    "pilot_3_cohort_size": len(pilot_3_pairs),
                    "pilot_10_assignments": [item.candidate_id for item in assignments],
                    "full_run_requested": False,
                    "seed_2_or_3_requested": False,
                    "scalar_hpo_enabled": False,
                },
            )
        )
        pilot_10_pairs = [item for item in pairs if item.stage_id == "pilot_10"]
        stages.append(_pilot_10_stage(pilot_10_pairs))
        stages.append(_policy_memory_stage(root, policy_memory_root, learning_results))
        if not reproduced:
            raise RuntimeError("no paper mechanism passed paired pilot_10")
        stages.append(
            _stage(
                "pilot_reproduced",
                artifacts=maturity_artifacts,
                metrics={
                    "component_ids": reproduced,
                    "next_boundary": "explicit_full_run_consent",
                    "full_training_started": False,
                    "multi_seed_started": False,
                },
            )
        )
        return _report(
            root=root,
            status="passed",
            model=model,
            device=device,
            research=research,
            objective_hash=objective_hash,
            protocol_hash=acceptance_protocol_hash,
            stages=stages,
            pairs=pairs,
            policy_memory_root=policy_memory_root,
            reproduced=reproduced,
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
        return _report(
            root=root,
            status="recovery",
            model=model,
            device=device,
            research=research,
            objective_hash=objective_hash,
            protocol_hash=acceptance_protocol_hash,
            stages=stages,
            pairs=pairs,
            policy_memory_root=policy_memory_root,
            recovery_actions=exc.actions,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _report(
            root=root,
            status="failed",
            model=model,
            device=device,
            research=research,
            objective_hash=objective_hash,
            protocol_hash=acceptance_protocol_hash,
            stages=stages,
            pairs=pairs,
            policy_memory_root=policy_memory_root,
            failures=[str(exc)],
            reproduced=reproduced,
        )


def _run_pair(
    *,
    root: Path,
    backend: GpuAcceptanceBackend,
    store: EvidenceStore,
    run_id: str,
    recipe: PaperAcceptanceRecipe,
    track: PaperAcceptanceTrackContext,
    stage_id: str,
    epochs: int,
    seed: int,
    base_protocol_hash: str,
    objective_hash: str,
    environment: dict[str, Any],
    data_yaml: Path,
    model: str,
    device: str,
) -> MultiMechanismPilotPair:
    protocol_hash = hash_payload(
        {
            "base_protocol_hash": base_protocol_hash,
            "component_id": recipe.component_id,
            "fidelity": stage_id,
        }
    )
    slug = _slug(recipe.component_id)
    control = backend.train(
        candidate_id=f"baseline_{recipe.component_id}_{stage_id}",
        node_id=f"baseline_{slug}_{stage_id}",
        data_yaml=data_yaml,
        model=model,
        workdir=root,
        device=device,
        epochs=epochs,
        seed=seed,
        protocol_hash=protocol_hash,
        overrides={},
        objective_hash=objective_hash,
    )
    candidate = backend.train(
        candidate_id=recipe.component_id,
        node_id=f"{slug}_{stage_id}",
        data_yaml=data_yaml,
        model=model,
        workdir=root,
        device=device,
        epochs=epochs,
        seed=seed,
        protocol_hash=protocol_hash,
        overrides=dict(recipe.adapter_options),
        objective_hash=objective_hash,
    )
    comparison = compare_paper_protocols(
        control.protocol_identity,
        candidate.protocol_identity,
    )
    if not comparison.matched:
        raise RuntimeError(
            f"{recipe.component_id} candidate/control protocol mismatch: "
            + ", ".join(sorted(comparison.mismatched_fields))
        )
    if control.protocol_identity is None:
        raise RuntimeError("backend omitted control protocol identity")
    expected = build_paper_protocol_identity(
        data_yaml=data_yaml,
        protocol_hash=protocol_hash,
        objective_hash=objective_hash,
        epochs=epochs,
        seed=seed,
        ultralytics_version=str(environment["ultralytics_version"]),
    )
    if control.protocol_identity != expected:
        raise RuntimeError("backend protocol identity differs from acceptance protocol")
    validate_component_runtime_artifacts(
        candidate,
        component_id=track.component_id,
        protocol_hash=protocol_hash,
    )
    try:
        control_eval = backend.evaluate(
            run=control,
            data_yaml=data_yaml,
            workdir=root,
            device=device,
        )
        candidate_eval = backend.evaluate(
            run=candidate,
            data_yaml=data_yaml,
            workdir=root,
            device=device,
        )
        identity = _evidence_identity(expected, stage_id)
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
            f"{recipe.component_id} {stage_id} COCO post-eval incomplete: {exc}",
            ["recover_control_coco_post_eval", "recover_candidate_coco_post_eval"],
        ) from exc
    evidence_path = root / "evidence_contracts" / f"{slug}_{stage_id}.yaml"
    evidence = validate_paper_pilot_evidence(
        store=store,
        run_id=run_id,
        stage_id=f"{recipe.component_id}:{stage_id}",
        protocol_hash=protocol_hash,
        control_run=control,
        control_evaluation=control_eval,
        candidate_run=candidate,
        candidate_evaluation=candidate_eval,
        output_path=evidence_path,
    )
    if not evidence.complete:
        raise PaperEvidenceRecoveryRequired(
            f"{recipe.component_id} {stage_id} evidence is incomplete: "
            + ", ".join(evidence.missing_evidence),
            evidence.recovery_actions,
        )
    bootstrap = _run_paired_bootstrap(
        data_yaml=data_yaml,
        control=control_eval,
        candidate=candidate_eval,
        output=root / "paired_bootstrap" / f"{slug}_{stage_id}.json",
        seed=seed + epochs,
    )
    if bootstrap.status != "completed":
        raise PaperEvidenceRecoveryRequired(
            f"{recipe.component_id} {stage_id} paired bootstrap is incomplete",
            ["recover_paired_bootstrap"],
        )
    _import_bootstrap_metrics(
        store,
        run_id=run_id,
        run=candidate,
        identity=_evidence_identity(expected, stage_id),
        report=bootstrap,
    )
    evidence_set = store.load_run(run_id)
    additional_metrics = list(
        dict.fromkeys([*recipe.target_metrics, "map50_95"])
    )
    paired = build_paired_experiment_result(
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        candidate_node_id=candidate.node_id,
        metric_records=evidence_set.metric_records,
        error_facts=ErrorFactStore(store.root).read(run_id),
        primary_metric=recipe.primary_metric,
        target_error_facts=recipe.target_error_facts,
        additional_metrics=additional_metrics,
    )
    if not paired.verified:
        raise PaperEvidenceRecoveryRequired(
            f"{recipe.component_id} {stage_id} paired result incomplete: "
            + ", ".join(paired.blockers),
            ["recover_verified_paired_delta"],
        )
    promotion, summary = evaluate_paper_recipe_promotion(
        recipe=recipe,
        stage_id=stage_id,
        paired=paired,
        control=control_eval,
        candidate=candidate_eval,
        bootstrap=bootstrap,
        paired_result_path=root / "paired_results" / f"{slug}_{stage_id}.json",
    )
    return MultiMechanismPilotPair(
        recipe=recipe,
        track=track,
        stage_id=stage_id,
        protocol=expected,
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


def _validated_tracks(
    research: PaperAcceptanceResearchContext,
) -> dict[str, PaperAcceptanceTrackContext]:
    tracks = {item.component_id: item for item in research.effective_tracks()}
    missing = [
        item.component_id
        for item in PAPER_ACCEPTANCE_RECIPES
        if item.component_id not in tracks
    ]
    families = {tracks[item.component_id].component_family for item in PAPER_ACCEPTANCE_RECIPES if item.component_id in tracks}
    if missing:
        raise RuntimeError("research context is missing tracks: " + ", ".join(missing))
    if len(families) != len(PAPER_ACCEPTANCE_RECIPES):
        raise RuntimeError("paper acceptance tracks must use distinct component families")
    return tracks


def _research_stages(
    research: PaperAcceptanceResearchContext,
    tracks: dict[str, PaperAcceptanceTrackContext],
    research_path: Path,
) -> list[PaperAutoOptimizationStage]:
    identities = {
        component_id: {
            "paper_ids": track.paper_ids,
            "method_profile_ids": track.method_profile_ids,
            "component_family": track.component_family,
            "adapter_hash": track.adapter_hash,
            "maturity": track.maturity,
        }
        for component_id, track in tracks.items()
    }
    return [
        _stage(
            "fresh_snapshot",
            artifacts={"research_context": research_path.as_posix()},
            metrics={"snapshot_hash": research.snapshot_hash},
        ),
        _stage(
            "diagnosis",
            metrics={
                "target_error_facts_by_component": {
                    item.component_id: item.target_error_facts
                    for item in PAPER_ACCEPTANCE_RECIPES
                },
                "paper_claim_used_as_local_evidence": False,
            },
        ),
        _stage("method_profile", metrics={"tracks": identities}),
        _stage(
            "certified_adapter",
            metrics={"tracks": identities, "scalar_hpo_enabled": False},
        ),
    ]


def _pilot_3_cohort_stages(
    root: Path,
    pairs: list[MultiMechanismPilotPair],
) -> list[PaperAutoOptimizationStage]:
    return [
        _stage(
            "matched_pilot_3_cohort",
            metrics={
                "cohort_size": len(pairs),
                "component_ids": [item.recipe.component_id for item in pairs],
                "component_families": [item.recipe.component_family for item in pairs],
                "all_protocols_matched": all(item.summary.protocol_match for item in pairs),
            },
        ),
        _stage(
            "candidate_control_post_eval",
            metrics={"stage": "pilot_3", "evaluated_nodes": len(pairs) * 2},
        ),
        _stage(
            "complete_coco_error_facts",
            artifacts={
                item.recipe.component_id: (
                    root
                    / "evidence_contracts"
                    / f"{_slug(item.recipe.component_id)}_pilot_3.yaml"
                ).as_posix()
                for item in pairs
            },
            metrics={
                "current_run_only": True,
                "current_node_only": True,
                "same_protocol_hash": True,
                "candidate_fact_count": sum(item.evidence.candidate_fact_count for item in pairs),
                "baseline_fact_count": sum(item.evidence.baseline_fact_count for item in pairs),
            },
        ),
        _stage(
            "paired_bootstrap_delta",
            metrics={
                item.recipe.component_id: item.summary.model_dump(mode="json")
                for item in pairs
            },
        ),
    ]


def _pilot_10_stage(
    pairs: list[MultiMechanismPilotPair],
) -> PaperAutoOptimizationStage:
    passed = bool(pairs) and all(item.promotion.passed for item in pairs)
    return _stage(
        "pilot_10",
        status="passed" if passed else "failed",
        message=(
            "paired pilot_10 survivors passed promotion"
            if passed
            else "pilot_10 survivor eliminated or absent"
        ),
        metrics={
            "results": {
                item.recipe.component_id: item.summary.model_dump(mode="json")
                for item in pairs
            },
            "full_training_started": False,
            "seed_2_or_3_started": False,
        },
    )


def _record_outcome(
    *,
    root: Path,
    policy_memory_root: Path | str,
    run_id: str,
    research: PaperAcceptanceResearchContext,
    pair_3: MultiMechanismPilotPair,
    pair_10: MultiMechanismPilotPair | None,
    failure_reason: str | None,
) -> PaperOutcomeLearningResult:
    fidelity = "pilot_10" if pair_10 is not None else "pilot_3"
    return record_paper_pilot_outcome(
        memory_root=policy_memory_root,
        run_id=run_id,
        research=research,
        track=pair_3.track,
        recipe=pair_3.recipe,
        protocol=(pair_10 or pair_3).protocol,
        pilot_3=pair_3.paired,
        pilot_10=pair_10.paired if pair_10 is not None else None,
        output_path=(
            root
            / "artifacts"
            / "policy_memory"
            / f"{_slug(pair_3.recipe.component_id)}_{fidelity}.json"
        ),
        failure_reason=failure_reason,
    )


def _policy_memory_stage(
    root: Path,
    policy_memory_root: Path | str,
    results: list[PaperOutcomeLearningResult],
) -> PaperAutoOptimizationStage:
    return _stage(
        "policy_memory",
        artifacts={
            "updates": (root / "artifacts" / "policy_memory").as_posix(),
            "policy_memory": (
                Path(policy_memory_root) / "policy_memory.jsonl"
            ).as_posix(),
        },
        metrics={
            "record_ids": [item.record.record_id for item in results],
            "failure_reasons": [
                item.record.failure_reason
                for item in results
                if item.record.failure_reason
            ],
            "eliminated_outcomes_recorded": any(
                item.record.failure_reason for item in results
            ),
            "paper_prior_is_local_evidence": False,
        },
    )


def _report(
    *,
    root: Path,
    status: str,
    model: str,
    device: str,
    research: PaperAcceptanceResearchContext | None,
    objective_hash: str,
    protocol_hash: str | None,
    stages: list[PaperAutoOptimizationStage],
    pairs: list[MultiMechanismPilotPair],
    policy_memory_root: Path | str,
    failures: list[str] | None = None,
    recovery_actions: list[str] | None = None,
    reproduced: list[str] | None = None,
) -> PaperAutoOptimizationReport:
    tracks = research.effective_tracks() if research is not None else []
    payload_hashes = []
    for pair in pairs:
        payload_path = pair.candidate_run.runtime_artifacts.get("runtime_payload")
        if payload_path is not None and payload_path.is_file():
            payload_hashes.append(AdapterRuntimePayload.read(payload_path).payload_hash)
    reproduced = reproduced or []
    return PaperAutoOptimizationReport(
        acceptance_id=root.name or "paper-auto-optimization",
        status=status,  # type: ignore[arg-type]
        execute_real_gpu=True,
        model=model,
        device=device,
        research_snapshot_hash=research.snapshot_hash if research else None,
        research_snapshot_path=research.snapshot_path if research else None,
        paper_ids=sorted({paper for item in tracks for paper in item.paper_ids}),
        component_ids=[item.component_id for item in tracks],
        component_families=[item.component_family for item in tracks],
        adapter_hash=(
            hash_payload({item.component_id: item.adapter_hash for item in tracks})
            if tracks
            else None
        ),
        maturity="pilot_reproduced" if reproduced else None,
        runtime_payload_hash=(
            hash_payload(sorted(set(payload_hashes))) if payload_hashes else None
        ),
        objective_hash=objective_hash,
        protocol_hash=protocol_hash,
        stages=stages,
        protocol_identities={
            f"{item.recipe.component_id}:{item.stage_id}": item.protocol
            for item in pairs
        },
        paired_deltas=[item.summary for item in pairs],
        asha_survivor=reproduced[0] if reproduced else None,
        asha_survivors=reproduced,
        policy_memory_path=Path(policy_memory_root) / "policy_memory.jsonl",
        pilot_reproduced=bool(reproduced),
        pilot_reproduced_component_ids=reproduced,
        evidence_recovery_actions=recovery_actions or [],
        failures=failures or [],
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


def _stage(
    stage_id: PaperAutoOptimizationStageId,
    *,
    status: str = "passed",
    message: str = "",
    artifacts: dict[str, str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> PaperAutoOptimizationStage:
    now = datetime.now(timezone.utc)
    return PaperAutoOptimizationStage(
        stage_id=stage_id,
        status=status,  # type: ignore[arg-type]
        message=message,
        artifacts=artifacts or {},
        metrics=metrics or {},
        started_at=now,
        completed_at=now,
    )


def _slug(value: str) -> str:
    return value.replace(".", "_").replace("/", "_")


__all__ = [
    "MultiMechanismPilotPair",
    "PaperEvidenceRecoveryRequired",
    "run_multi_mechanism_acceptance",
]
