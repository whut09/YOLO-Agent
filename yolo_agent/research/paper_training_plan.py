"""Materialize the current real paper cohort without executing training.

This module is deliberately narrower than the research planner.  It consumes
already-produced production artifacts, prepares only candidates that have
passed the real paper readiness report, and writes a paired baseline/candidate
queue plus ASHA identities.  It never creates evidence or promotes maturity.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from yolo_agent.agents.asha_scheduler import ASHAScheduler, ASHAStudy
from yolo_agent.agents.candidate_generator import CandidateConfig, CandidateEvaluationContract
from yolo_agent.certification.paper_readiness import PaperReadinessReport
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.execution_bridge import ComponentExecutionBridge
from yolo_agent.components.maturity_registry import ComponentMaturityRegistry
from yolo_agent.core.command_spec import CommandSpec
from yolo_agent.core.execution_queue import ExecutionQueue
from yolo_agent.core.experiment_graph import ExperimentNode
from yolo_agent.core.round_execution_plan import (
    RoundAblationNode,
    RoundExecutionPlan,
    RoundStageSpec,
    build_asha_assignment_plan,
)
from yolo_agent.core.run_context import RunContext
from yolo_agent.research.paper_asset_dependencies import (
    requires_domain_assets,
    requires_hard_negative_replay,
    requires_teacher_checkpoint,
)
from yolo_agent.research.paper_asset_schemas import PaperAssetRegistry
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
)
from yolo_agent.research.paper_training_plan_schemas import (
    PaperTrainingPlan,
    PaperTrainingPlanRecord,
)
from yolo_agent.resources import ResourcePaths
from yolo_agent.recipes.registry import RecipeRegistry


class PaperTrainingPlanBuilder:
    """Build a dry-run queue and ASHA study for currently eligible papers."""

    def build(
        self,
        *,
        run_id: str,
        run_root: Path | str = "runs",
        inventory_path: Path | str = Path(
            "runs/coverage-audit/paper_execution_inventory.yaml"
        ),
        requirements_path: Path | str = Path(
            "runs/coverage-audit/paper_execution_requirements.yaml"
        ),
        assets_path: Path | str = Path(
            "runs/paper-readiness/paper_asset_registry.yaml"
        ),
        readiness_path: Path | str = Path(
            "runs/paper-readiness/paper_readiness_report.yaml"
        ),
        model: str = "yolo26n.pt",
        data: Path | str = Path(r"E:\datatset\coco.yaml"),
        output_path: Path | str | None = None,
        maturity_registry_path: Path | str | None = Path(
            "runs/component_maturity_registry.yaml"
        ),
    ) -> PaperTrainingPlan:
        run_dir = (Path(run_root) / run_id).resolve()
        context_path = run_dir / "run_context.yaml"
        if not context_path.is_file():
            raise ValueError(
                "paper cohort preparation requires an initialized run context: "
                f"{context_path}"
            )
        context = RunContext.from_yaml(context_path)
        if not context.run_protocol_hash or len(context.run_protocol_hash) != 64:
            raise ValueError("initialized run has no valid run protocol hash")

        inventory_file = _existing_file(inventory_path, "inventory")
        requirements_file = _existing_file(requirements_path, "requirements")
        assets_file = _existing_file(assets_path, "assets")
        readiness_file = _existing_file(readiness_path, "readiness")
        inventory = PaperExecutionInventory.from_yaml(inventory_file)
        requirements = PaperExecutionRequirementsMatrix.from_yaml(requirements_file)
        assets = PaperAssetRegistry.from_yaml(assets_file)
        readiness = PaperReadinessReport.from_yaml(readiness_file)
        _validate_inputs(
            inventory=inventory,
            requirements=requirements,
            assets=assets,
            readiness=readiness,
        )

        requirements_by_id = {item.paper_id: item for item in requirements.requirements}
        assets_by_id = {item.paper_id: item for item in assets.records}
        readiness_by_id = {item.paper_id: item for item in readiness.records}
        selected = _select_rows(
            inventory.records,
            requirements_by_id,
            assets_by_id,
            readiness_by_id,
        )
        if not selected:
            raise ValueError(
                "no production paper execution has readiness=asha_eligible and "
                "a training route"
            )

        recipes = RecipeRegistry.from_paths(
            sorted(ResourcePaths.RECIPES_DIR.glob("*.yaml")),
            strict=False,
        )
        contracts = _load_selected_contracts(
            selected,
            maturity_registry_path=maturity_registry_path,
        )
        bridge = ComponentExecutionBridge()
        plan_candidates: list[PaperTrainingPlanRecord] = []
        paired_plans: list[tuple[ExperimentNode, ExperimentNode, PaperTrainingPlanRecord]] = []
        for index, group in enumerate(_group_rows(selected), start=1):
            item, requirement, asset, preflight = group[0]
            recipe = recipes.get(requirement.recipe_ids[0]) if requirement.recipe_ids else None
            if recipe is None:
                raise ValueError(
                    f"paper {item.paper_id} references an unavailable recipe: "
                    f"{requirement.recipe_ids}"
                )
            component_contracts = {
                component_id: contracts[component_id]
                for component_id in recipe.component_ids
                if component_id in contracts
            }
            if set(component_contracts) != set(recipe.component_ids):
                missing = sorted(set(recipe.component_ids) - set(component_contracts))
                raise ValueError(
                    f"paper {item.paper_id} has no loaded executable contract: {missing}"
                )
            paper_ids = sorted(row[0].paper_id for row in group)
            profile_ids = sorted(row[0].profile_id for row in group)
            protocol_hash = str(preflight.protocol_hash)
            dataset_hash = str(preflight.dataset_manifest_hash)
            candidate_id = f"paper_cohort_{index:02d}_{_safe_name(item.paper_id)}"
            base_candidate = _candidate_node(
                run_dir=run_dir,
                candidate_id=candidate_id,
                paper_ids=paper_ids,
                profile_ids=profile_ids,
                item=item,
                requirement=requirement,
                recipe=recipe,
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                model=model,
                data=Path(asset.source_dataset_manifest).resolve(),
            )
            baseline = _baseline_node(
                run_dir=run_dir,
                baseline_id=f"matched_baseline_{index:02d}_{_safe_name(item.paper_id)}",
                protocol_hash=protocol_hash,
                dataset_hash=dataset_hash,
                model=model,
                data=Path(asset.source_dataset_manifest).resolve(),
            )
            runtime = bridge.prepare(
                recipe=recipe,
                node=base_candidate,
                contracts=component_contracts,
                workspace=run_dir / "artifacts" / "paper_runtime" / candidate_id,
                protocol_hash=protocol_hash,
                dry_run=True,
            )
            if runtime.status != "executable" or runtime.runtime_payload_path is None:
                raise ValueError(
                    f"paper {item.paper_id} adapter preparation failed: "
                    + ",".join(runtime.blocked_by)
                )
            candidate = runtime.node
            single_plan = build_asha_assignment_plan(
                run_id=run_id,
                source_node=candidate,
                stage_id="pilot_3",
                epochs=3,
                fraction=0.1,
                seed=1,
                run_name=candidate_id,
                baseline_control_node=baseline,
                primary_metric="map50_95",
                assignment_id=f"paper-cohort:{run_id}:{item.execution_fingerprint}:pilot_3",
            )
            control_source = next(
                node
                for node in single_plan.deferred_nodes
                if node.command_spec is not None
                and node.command_spec.metadata.get("matched_baseline_control")
            )
            plan_record = PaperTrainingPlanRecord(
                paper_ids=paper_ids,
                profile_ids=profile_ids,
                mechanism_ids=sorted(set(item.paper_specific_mechanism_ids)),
                recipe_id=recipe.recipe_id,
                recipe_version=recipe.version,
                execution_fingerprint=item.execution_fingerprint,
                candidate_node_id=candidate.node_id,
                baseline_node_id=control_source.node_id,
                asha_trial_id=_trial_id(run_id, item.execution_fingerprint),
                protocol_hash=protocol_hash,
                dataset_manifest_hash=dataset_hash,
                runtime_payload_path=runtime.runtime_payload_path.resolve().as_posix(),
                runtime_payload_hash=runtime.runtime_payload_hash or "",
                candidate_command=[str(value) for value in candidate.command_spec.argv],
                baseline_command=[str(value) for value in control_source.command_spec.argv],
            )
            plan_candidates.append(plan_record)
            paired_plans.append((candidate, control_source, plan_record))

        round_plan = _merge_round_plans(
            run_id=run_id,
            paired_plans=paired_plans,
            requirements_by_id=requirements_by_id,
            inventory=inventory,
        )
        round_plan_path = run_dir / "artifacts" / "round_execution_plan.yaml"
        round_plan.to_yaml(round_plan_path, exclude_none=True, sort_keys=False)
        projection = round_plan.experiment_projection()
        projection_path = run_dir / "artifacts" / "experiment_plan.yaml"
        projection.to_yaml(projection_path, exclude_none=True, sort_keys=False)
        queue = ExecutionQueue.from_round_execution_plan(run_id, round_plan)
        queue.metadata.update(
            {
                "paper_training_cohort": True,
                "dry_run_prepared": True,
                "paper_inventory_count": inventory.compatible_paper_count,
                "trainable_fingerprint_count": len(plan_candidates),
                "has_baseline_controls": True,
                "has_paper_candidates": True,
            }
        )
        queue_path = run_dir / "execution_queue.yaml"
        queue.to_yaml(queue_path, exclude_none=True, sort_keys=False)

        asha_path = run_dir / "artifacts" / "asha_state.yaml"
        study = ASHAStudy.from_yaml(asha_path) if asha_path.is_file() else ASHAScheduler.create(run_id).study
        study.run_protocol_hash = context.run_protocol_hash
        study.metadata.update(
            {
                "paper_training_cohort": True,
                "paper_cohort_per_candidate_protocols": True,
                "dry_run_queue_required": True,
                "paper_training_plan_path": (run_dir / "artifacts" / "paper_training_plan.yaml").as_posix(),
                "paper_inventory_path": inventory_file.as_posix(),
                "paper_inventory_hash": inventory.inventory_hash,
            }
        )
        scheduler = ASHAScheduler(study)
        for candidate, control, record in paired_plans:
            item = next(row[0] for row in selected if row[0].execution_fingerprint == record.execution_fingerprint)
            requirement = requirements_by_id[item.paper_id]
            scheduler.register_trial(
                trial_id=record.asha_trial_id,
                candidate_id=candidate.candidate_config.candidate_id,
                source_run_id=run_id,
                source_node=candidate,
                baseline_control_node=control,
                target_error_facts=candidate.candidate_config.target_error_facts,
                paper_ids=record.paper_ids,
                method_profile_ids=record.profile_ids,
                mechanism_ids=record.mechanism_ids,
                required_evidence=list(requirement.required_evidence),
                readiness_state="asha_eligible",
                execution_fingerprint_override=record.execution_fingerprint,
                paper_specific_configuration={
                    "paper_ids": record.paper_ids,
                    "mechanism_ids": record.mechanism_ids,
                    "recipe_id": record.recipe_id,
                    "recipe_version": record.recipe_version,
                    "protocol_hash": record.protocol_hash,
                },
            )
        scheduler.study.to_yaml(asha_path, exclude_none=True, sort_keys=False)

        context.metadata.update(
            {
                "paper_training_cohort_prepared": True,
                "paper_training_cohort": True,
                "paper_training_cohort_plan_path": (run_dir / "artifacts" / "paper_training_plan.yaml").as_posix(),
                "paper_training_cohort_queue_path": queue_path.as_posix(),
                "paper_training_cohort_asha_path": asha_path.as_posix(),
                "paper_training_cohort_inventory_hash": inventory.inventory_hash,
                "paper_training_cohort_fingerprints": ",".join(
                    sorted(record.execution_fingerprint for record in plan_candidates)
                ),
            }
        )
        context.to_yaml()
        context.to_json()
        output = Path(output_path) if output_path is not None else run_dir / "artifacts" / "paper_training_plan.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        plan = PaperTrainingPlan(
            run_id=run_id,
            run_dir=run_dir.as_posix(),
            inventory_path=inventory_file.as_posix(),
            requirements_path=requirements_file.as_posix(),
            assets_path=assets_file.as_posix(),
            readiness_path=readiness_file.as_posix(),
            asha_path=asha_path.as_posix(),
            round_plan_path=round_plan_path.as_posix(),
            queue_path=queue_path.as_posix(),
            run_protocol_hash=context.run_protocol_hash,
            total_papers=inventory.compatible_paper_count,
            trainable_fingerprints=len(plan_candidates),
            matched_controls_planned=len(plan_candidates),
            asha_trials_registered=len(plan_candidates),
            records=plan_candidates,
            training_allowed=True,
        ).with_hash()
        plan.to_yaml(output, exclude_none=True, sort_keys=False)
        return plan


def build_paper_training_plan(**kwargs: object) -> PaperTrainingPlan:
    """Functional entrypoint for the offline preparation command."""

    return PaperTrainingPlanBuilder().build(**kwargs)  # type: ignore[arg-type]


def _select_rows(records: list[Any], requirements: dict[str, Any], assets: dict[str, Any], readiness: dict[str, Any]) -> list[tuple[Any, Any, Any, Any]]:
    selected: list[tuple[Any, Any, Any, Any]] = []
    for item in records:
        requirement = requirements[item.paper_id]
        asset = assets[item.paper_id]
        preflight = readiness[item.paper_id]
        mechanisms = set(item.paper_specific_mechanism_ids) | set(requirement.paper_specific_mechanism_ids)
        payload = requirement.required_runtime_payload
        if not preflight.asha_eligibility or preflight.readiness_state != "asha_eligible":
            continue
        if requirement.execution_route == "inference" or not requirement.required_adapter:
            continue
        if not requirement.required_changed_variables or not payload:
            continue
        if str(payload.get("mode") or "") == "shadow":
            continue
        if not requirement.compatible_with_yolo26 or asset.availability != "available":
            continue
        if requires_teacher_checkpoint(mechanisms) or requires_domain_assets(mechanisms) or requires_hard_negative_replay(mechanisms):
            continue
        if _contains_mock(item.model_dump(mode="json")) or _contains_mock(requirement.model_dump(mode="json")) or _contains_mock(asset.model_dump(mode="json")) or _contains_mock(preflight.model_dump(mode="json")):
            continue
        selected.append((item, requirement, asset, preflight))
    return selected


def _group_rows(rows: list[tuple[Any, Any, Any, Any]]) -> list[list[tuple[Any, Any, Any, Any]]]:
    grouped: dict[str, list[tuple[Any, Any, Any, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[0].execution_fingerprint].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _candidate_node(*, run_dir: Path, candidate_id: str, paper_ids: list[str], profile_ids: list[str], item: Any, requirement: Any, recipe: Any, protocol_hash: str, dataset_hash: str, model: str, data: Path) -> ExperimentNode:
    metadata: dict[str, str | int | float | bool] = {
        "paper_ids": ",".join(paper_ids),
        "method_profile_ids": ",".join(profile_ids),
        "paper_execution_fingerprint": item.execution_fingerprint,
        "adapter_runtime_entrypoint": "pending_component_bridge",
        "component_recipe_id": recipe.recipe_id,
        "component_recipe_version": recipe.version,
        "run_protocol_hash": protocol_hash,
        "baseline_protocol_hash": protocol_hash,
        "protocol_hash": protocol_hash,
        "dataset_manifest_hash": dataset_hash,
        "dataset_manifest_sha256": dataset_hash,
        "fidelity": "pilot_3",
        "round_stage": "pilot_3",
        "training_budget_profile": "pilot",
        "seed_policy": "fixed_seed_1",
        "split": "coco_val",
        "evaluation_split": "coco_val",
        "imgsz": 640,
        "model": model,
        "paper_readiness_state": "asha_eligible",
        "paper_readiness_blockers": "[]",
        "matched_control_plan_required": True,
        "training_attribution_allowed": True,
    }
    command = CommandSpec.ultralytics_train(
        model=model,
        data=data,
        project=run_dir / "ultralytics",
        name=candidate_id,
        epochs=3,
        imgsz=640,
        batch=32,
        seed=1,
        workers=8,
        amp=True,
        metadata=metadata,
    )
    evaluation = CandidateEvaluationContract(
        primary_metric="map50_95",
        evaluation_metrics=list(recipe.target_metrics),
        stop_conditions=list(recipe.stop_conditions),
        promotion_requirements=list(recipe.promotion_requirements),
    )
    changed = {recipe.primary_changed_variable: recipe.train_overrides.get(recipe.primary_changed_variable, 0.1)}
    return ExperimentNode(
        node_id=f"node_{candidate_id}",
        candidate_config=CandidateConfig(
            candidate_id=candidate_id,
            base_model=model,
            scale="n",
            framework="ultralytics",
            components=list(recipe.component_ids),
            train_overrides=dict(recipe.train_overrides),
            action_domain="paper",
            action_id=recipe.recipe_id,
            search_tier="method",
            target_error_facts=list(recipe.target_error_facts),
            evaluation_contract=evaluation,
            expected_effect=[str(key) for key in recipe.expected_effects],
            risk=recipe.implementation_risk if recipe.implementation_risk in {"low", "medium", "high"} else "low",
        ),
        data_version=dataset_hash,
        seed=1,
        command=command.display(),
        command_spec=command,
        fixed_variables=dict(recipe.fixed_variables),
        changed_variables=changed,
    )


def _baseline_node(*, run_dir: Path, baseline_id: str, protocol_hash: str, dataset_hash: str, model: str, data: Path) -> ExperimentNode:
    metadata: dict[str, str | int | float | bool] = {
        "matched_baseline_control": True,
        "paper_cohort_baseline": True,
        "run_protocol_hash": protocol_hash,
        "baseline_protocol_hash": protocol_hash,
        "protocol_hash": protocol_hash,
        "dataset_manifest_hash": dataset_hash,
        "dataset_manifest_sha256": dataset_hash,
        "fidelity": "pilot_3",
        "round_stage": "pilot_3",
        "training_budget_profile": "pilot",
        "seed_policy": "fixed_seed_1",
        "split": "coco_val",
        "evaluation_split": "coco_val",
        "imgsz": 640,
        "model": model,
        "paper_readiness_state": "pre_registered",
        "paper_readiness_blockers": "[]",
        "training_attribution_allowed": False,
    }
    command = CommandSpec.ultralytics_train(
        model=model,
        data=data,
        project=run_dir / "ultralytics",
        name=baseline_id,
        epochs=3,
        imgsz=640,
        batch=32,
        seed=1,
        workers=8,
        amp=True,
        metadata=metadata,
    )
    return ExperimentNode(
        node_id=f"node_{baseline_id}",
        candidate_config=CandidateConfig(
            candidate_id=baseline_id,
            base_model=model,
            scale="n",
            framework="ultralytics",
            action_domain="baseline",
            action_id="matched-baseline",
            search_tier="method",
        ),
        data_version=dataset_hash,
        seed=1,
        command=command.display(),
        command_spec=command,
    )


def _merge_round_plans(*, run_id: str, paired_plans: list[tuple[ExperimentNode, ExperimentNode, PaperTrainingPlanRecord]], requirements_by_id: dict[str, Any], inventory: PaperExecutionInventory) -> RoundExecutionPlan:
    execution_nodes: list[ExperimentNode] = []
    assignments: list[Any] = []
    deferred_nodes: list[ExperimentNode] = []
    matched: dict[str, Any] = {}
    ablations: list[RoundAblationNode] = []
    evidence_requirements: dict[str, list[str]] = {}
    selected_recipes: list[dict[str, Any]] = []
    for candidate, control, record in paired_plans:
        single = build_asha_assignment_plan(
            run_id=run_id,
            source_node=candidate,
            stage_id="pilot_3",
            epochs=3,
            fraction=0.1,
            seed=1,
            run_name=candidate.candidate_config.candidate_id,
            baseline_control_node=control,
            primary_metric="map50_95",
            assignment_id=f"paper-cohort:{run_id}:{record.execution_fingerprint}:pilot_3",
        )
        execution_nodes.extend(single.execution_nodes)
        assignments.extend(single.assignments)
        for node in single.deferred_nodes:
            if node.node_id not in {item.node_id for item in deferred_nodes}:
                deferred_nodes.append(node)
        matched.update(single.matched_control_plans)
        ablations.append(
            RoundAblationNode(
                node_id=candidate.node_id,
                candidate_id=candidate.candidate_config.candidate_id,
                changed_variables=dict(candidate.changed_variables),
                component_ids=list(candidate.candidate_config.components),
                guard_metrics=["map50_95", "latency_ms", "model_size_mb"],
                reason="paper_cohort_atomic_recipe",
            )
        )
        paper_id = record.paper_ids[0]
        evidence_requirements[candidate.node_id] = list(requirements_by_id[paper_id].required_evidence)
        selected_recipes.append(
            {
                "recipe_id": record.recipe_id,
                "version": record.recipe_version,
                "paper_ids": record.paper_ids,
                "execution_fingerprint": record.execution_fingerprint,
            }
        )
    return RoundExecutionPlan(
        run_id=run_id,
        round_id=f"{run_id}_paper_cohort",
        stages=[
            RoundStageSpec(stage_id="pilot_3", training_profile="pilot", epochs=3, fraction=0.1, keep_ratio=0.5),
            RoundStageSpec(stage_id="pilot_10", training_profile="pilot", epochs=10, fraction=0.1, keep_top_k=2),
            RoundStageSpec(stage_id="candidate_full_seed_1", training_profile="candidate_full", epochs=100, fraction=1.0, keep_top_k=1),
            RoundStageSpec(stage_id="candidate_full_confirmation", training_profile="candidate_full", epochs=100, fraction=1.0, keep_top_k=1),
        ],
        assignments=assignments,
        ablation_nodes=ablations,
        execution_nodes=execution_nodes,
        deferred_nodes=deferred_nodes,
        evidence_requirements=evidence_requirements,
        matched_control_plans=matched,
        selected_recipes=selected_recipes,
        active_stage="pilot_3",
        primary_metric="map50_95",
        scheduler_mode="round_halving",
        status="ready",
    )


def _load_selected_contracts(rows: list[tuple[Any, Any, Any, Any]], *, maturity_registry_path: Path | str | None) -> dict[str, ComponentContract]:
    component_ids = {
        component_id
        for _, requirement, _, _ in rows
        for component_id in requirement.paper_specific_mechanism_ids
    }
    registry = ComponentMaturityRegistry(maturity_registry_path) if maturity_registry_path and Path(maturity_registry_path).is_file() else None
    contracts: dict[str, ComponentContract] = {}
    for path in sorted(ResourcePaths.COMPONENTS_DIR.rglob("*.yaml")):
        try:
            loaded = load_contracts(path, maturity_registry=registry)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        for contract in loaded:
            if contract.component_id in component_ids:
                contracts[contract.component_id] = contract
    return contracts


def _validate_inputs(*, inventory: PaperExecutionInventory, requirements: PaperExecutionRequirementsMatrix, assets: PaperAssetRegistry, readiness: PaperReadinessReport) -> None:
    if inventory.compatible_paper_count != 83:
        raise ValueError(f"expected 83 inventory papers, got {inventory.compatible_paper_count}")
    ids = {item.paper_id for item in inventory.records}
    if ids != {item.paper_id for item in requirements.requirements} or ids != {item.paper_id for item in assets.records} or ids != {item.paper_id for item in readiness.records}:
        raise ValueError("paper cohort inputs do not have identical 83-paper coverage")
    if requirements.source_inventory_hash != inventory.inventory_hash:
        raise ValueError("requirements are stale relative to inventory")
    if assets.source_inventory_hash != inventory.inventory_hash:
        raise ValueError("assets are stale relative to inventory")


def _existing_file(path: Path | str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} artifact does not exist: {resolved}")
    return resolved


def _trial_id(run_id: str, fingerprint: str) -> str:
    return f"paper-cohort:{run_id}:{fingerprint}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value) or "paper"


def _contains_mock(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_mock(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_mock(item) for item in value)
    if not isinstance(value, str):
        return False
    text = value.lower()
    return any(marker in text for marker in ("mock", "fixture", "pytest_fixture", "offline_mock"))


__all__ = ["PaperTrainingPlanBuilder", "build_paper_training_plan"]
