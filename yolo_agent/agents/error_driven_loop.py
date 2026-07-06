"""Error-driven optimization loop orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from yolo_agent.agents.augmentation_policy import AugmentationPolicyEngine, AugmentationPolicyResult
from yolo_agent.agents.error_to_action import ActionPolicy, DetectionErrorObservation, ErrorActionMapper, ErrorActionPlan
from yolo_agent.agents.optimization_recipe import OptimizationRecipeEngine, OptimizationRecipePlan
from yolo_agent.agents.sampling_policy import SamplingPolicyEngine, SamplingPolicyPlan
from yolo_agent.agents.strategy_policy import CandidatePolicy, PolicyConstraint
from yolo_agent.components.postprocess import PostProcessRecommendation, PostProcessRegistry
from yolo_agent.core.schemas import DeploymentConstraints
from yolo_agent.core.task_spec import TaskSpec
from yolo_agent.tools.dataset_stats import DatasetReport
from yolo_agent.utils import dedupe_list


DiagnosisCategory = Literal[
    "data",
    "annotation",
    "model_capacity",
    "loss_assigner_head",
    "postprocess",
    "deployment",
]


class ClosedLoopDiagnosis(BaseModel):
    """One structured diagnosis with actionability."""

    category: DiagnosisCategory
    question: str
    answer: str
    supporting_signals: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class NextRoundPlan(BaseModel):
    """Single-variable next-round experiment guidance."""

    candidate_policies: list[CandidatePolicy] = Field(default_factory=list)
    changed_variables: dict[str, list[str]] = Field(default_factory=dict)
    evidence_required: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)


class ErrorDrivenLoopReport(BaseModel):
    """Full diagnosis-to-next-round report."""

    task_scene: str
    diagnostics: list[ClosedLoopDiagnosis]
    action_policy: ErrorActionPlan
    optimization_recipes: OptimizationRecipePlan
    sampling_policy: SamplingPolicyPlan
    augmentation_policy: AugmentationPolicyResult
    postprocess_policy: PostProcessRecommendation
    next_round: NextRoundPlan
    evidence_status: dict[str, str] = Field(default_factory=dict)


class ErrorDrivenLoopEngine:
    """Compose diagnosis, action policy, candidate proposals, and evidence needs."""

    def __init__(
        self,
        action_mapper: ErrorActionMapper | None = None,
        recipe_engine: OptimizationRecipeEngine | None = None,
        sampling_engine: SamplingPolicyEngine | None = None,
        augmentation_engine: AugmentationPolicyEngine | None = None,
        postprocess_registry: PostProcessRegistry | None = None,
    ) -> None:
        self.action_mapper = action_mapper or ErrorActionMapper.from_yaml()
        self.recipe_engine = recipe_engine or OptimizationRecipeEngine.from_yaml()
        self.sampling_engine = sampling_engine or SamplingPolicyEngine()
        self.augmentation_engine = augmentation_engine or AugmentationPolicyEngine.from_yaml()
        self.postprocess_registry = postprocess_registry or PostProcessRegistry.from_yaml()

    def run(
        self,
        task_spec: TaskSpec,
        dataset_report: DatasetReport,
        detection_errors: list[DetectionErrorObservation],
        deployment: DeploymentConstraints | None = None,
        evidence_status: dict[str, str] | None = None,
        fixed_imgsz: int | None = None,
    ) -> ErrorDrivenLoopReport:
        """Run the diagnosis-to-next-round optimization loop."""
        action_policy = self.action_mapper.map_errors(detection_errors)
        recipe_plan = self.recipe_engine.recommend(task_spec, detection_errors, dataset_report)
        sampling_policy = self.sampling_engine.recommend(dataset_report, detection_errors)
        augmentation_policy = self.augmentation_engine.recommend(dataset_report, detection_errors)
        postprocess_policy = self.postprocess_registry.recommend_for_errors(detection_errors, task_spec)
        diagnostics = _diagnose(
            task_spec=task_spec,
            dataset_report=dataset_report,
            detection_errors=detection_errors,
            deployment=deployment,
            recipe_plan=recipe_plan,
            sampling_policy=sampling_policy,
            postprocess_policy=postprocess_policy,
        )
        next_round = _next_round_plan(
            task_spec=task_spec,
            action_policy=action_policy,
            recipe_plan=recipe_plan,
            sampling_policy=sampling_policy,
            augmentation_policy=augmentation_policy,
            postprocess_policy=postprocess_policy,
            deployment=deployment,
            fixed_imgsz=fixed_imgsz,
        )
        return ErrorDrivenLoopReport(
            task_scene=task_spec.scene,
            diagnostics=diagnostics,
            action_policy=action_policy,
            optimization_recipes=recipe_plan,
            sampling_policy=sampling_policy,
            augmentation_policy=augmentation_policy,
            postprocess_policy=postprocess_policy,
            next_round=next_round,
            evidence_status=evidence_status or _default_evidence_status(next_round.evidence_required),
        )


def _diagnose(
    task_spec: TaskSpec,
    dataset_report: DatasetReport,
    detection_errors: list[DetectionErrorObservation],
    deployment: DeploymentConstraints | None,
    recipe_plan: OptimizationRecipePlan,
    sampling_policy: SamplingPolicyPlan,
    postprocess_policy: PostProcessRecommendation,
) -> list[ClosedLoopDiagnosis]:
    diagnostics: list[ClosedLoopDiagnosis] = []
    problems = set(dataset_report.dataset_health.problems)
    error_types = {observation.error_type for observation in detection_errors}

    if problems:
        diagnostics.append(
            ClosedLoopDiagnosis(
                category="data",
                question="Is the failure caused by dataset composition?",
                answer="Likely contributing factor: dataset health has actionable problems.",
                supporting_signals=sorted(problems),
                next_actions=[action.action_type for action in sampling_policy.actions],
                expected_metrics=["precision", "recall", "validation_stability"],
                risks=["Dataset changes can invalidate comparisons unless dataset version is recorded."],
            )
        )

    if "annotation_noise" in problems or recipe_plan.data_checks:
        diagnostics.append(
            ClosedLoopDiagnosis(
                category="annotation",
                question="Is the failure caused by labels?",
                answer="Label quality must be checked before trusting component ablations.",
                supporting_signals=dedupe_list([*recipe_plan.data_checks, *dataset_report.potential_issues]),
                next_actions=dedupe_list(["run_annotation_advisor", *recipe_plan.data_checks]),
                expected_metrics=["label_quality_report", "map50_95"],
                risks=["Loss changes may appear helpful when they are only fitting noisy boxes."],
            )
        )

    if error_types.intersection({"small_object_miss", "occlusion_miss", "out_of_distribution_miss"}):
        diagnostics.append(
            ClosedLoopDiagnosis(
                category="model_capacity",
                question="Is the failure caused by model capacity or feature resolution?",
                answer="Miss errors suggest checking feature resolution and capacity after data/label checks.",
                supporting_signals=sorted(error_types.intersection({"small_object_miss", "occlusion_miss", "out_of_distribution_miss"})),
                next_actions=dedupe_list([*recipe_plan.component_candidates.head, "compare_nano_vs_small_scale"]),
                expected_metrics=["recall", "mAP_small", "latency_ms"],
                risks=["Increasing feature resolution or scale can violate deployment latency."],
            )
        )

    if recipe_plan.component_candidates.all_ids():
        diagnostics.append(
            ClosedLoopDiagnosis(
                category="loss_assigner_head",
                question="Which train-time component should change next?",
                answer="Use the matched recipe and change one main variable per experiment.",
                supporting_signals=[recommendation.recipe_id for recommendation in recipe_plan.recommendations],
                next_actions=recipe_plan.component_candidates.all_ids(),
                expected_metrics=recipe_plan.evidence_required,
                risks=recipe_plan.risks,
            )
        )

    if postprocess_policy.ids:
        diagnostics.append(
            ClosedLoopDiagnosis(
                category="postprocess",
                question="Can inference policy explain part of the error?",
                answer="Post-processing should be calibrated before assuming the network is wrong.",
                supporting_signals=postprocess_policy.ids,
                next_actions=dedupe_list([*postprocess_policy.ids, *postprocess_policy.companion_actions]),
                expected_metrics=["precision", "recall", "latency_ms"],
                risks=postprocess_policy.warnings,
            )
        )

    if deployment is not None:
        diagnostics.append(
            ClosedLoopDiagnosis(
                category="deployment",
                question="Do deployment constraints limit the action space?",
                answer=_deployment_answer(task_spec, deployment),
                supporting_signals=_deployment_signals(task_spec, deployment),
                next_actions=["check_latency_budget", "check_export_runtime"],
                expected_metrics=["latency_ms", "model_size_mb", "fps"],
                risks=["High-latency recipes such as SAHI/TTA may be unsuitable for strict edge targets."],
            )
        )

    return diagnostics


def _next_round_plan(
    task_spec: TaskSpec,
    action_policy: ErrorActionPlan,
    recipe_plan: OptimizationRecipePlan,
    sampling_policy: SamplingPolicyPlan,
    augmentation_policy: AugmentationPolicyResult,
    postprocess_policy: PostProcessRecommendation,
    deployment: DeploymentConstraints | None,
    fixed_imgsz: int | None = None,
) -> NextRoundPlan:
    policies: list[CandidatePolicy] = []
    changed_variables: dict[str, list[str]] = {}
    guardrails: list[str] = []

    for component_type, component_ids in [
        ("bbox_loss", recipe_plan.component_candidates.bbox_loss),
        ("head_component", recipe_plan.component_candidates.head),
        ("assigner", recipe_plan.component_candidates.assigner),
    ]:
        for component_id in component_ids:
            policies.append(
                _policy(
                    policy_id=f"next_{component_type}_{_slug(component_id)}",
                    task_spec=task_spec,
                    components=[component_id],
                    train_overrides={},
                    action_domain="model",
                    action_id=component_id,
                    expected_effect=recipe_plan.expected_effect,
                    rationale=f"Single-variable test for {component_type}: {component_id}.",
                    deployment=deployment,
                )
            )
            changed_variables.setdefault(component_type, []).append(component_id)

    if "imgsz" in recipe_plan.train_overrides:
        value = recipe_plan.train_overrides["imgsz"]
        if _imgsz_change_allowed(value, fixed_imgsz):
            policies.append(
                _policy(
                    policy_id=f"next_imgsz_{value}",
                    task_spec=task_spec,
                    components=[],
                    train_overrides={"imgsz": value},
                    action_domain="training",
                    action_id="imgsz",
                    expected_effect=["Measure whether input resolution changes improve the observed error."],
                    rationale="Single-variable test for input resolution.",
                    deployment=deployment,
                )
            )
            changed_variables.setdefault("imgsz", []).append(str(value))
        else:
            guardrails.append(
                f"blocked_imgsz_increase: requested imgsz={value} exceeds fixed baseline imgsz={fixed_imgsz}; "
                "keep input size fixed for fair COCO/YOLO26 comparison."
            )

    for sampling_action in sampling_policy.actions:
        policies.append(
            _policy(
                policy_id=f"next_data_{_slug(sampling_action.action_type)}",
                task_spec=task_spec,
                components=[],
                train_overrides={
                    "data_action": sampling_action.action_type,
                    "sampling_target": sampling_action.target,
                    "sampling_parameters": sampling_action.parameters,
                },
                action_domain="data",
                action_id=sampling_action.action_type,
                expected_effect=[sampling_action.rationale],
                rationale=f"Single-variable test for data action: {sampling_action.action_type}.",
                deployment=deployment,
                priority_hint=max(1.0, sampling_action.priority / 5.0),
            )
        )
        changed_variables.setdefault("data_action", []).append(sampling_action.action_type)

    label_actions = _label_actions(action_policy, recipe_plan.data_checks)
    for label_action in label_actions:
        policies.append(
            _policy(
                policy_id=f"next_label_{_slug(label_action)}",
                task_spec=task_spec,
                components=[],
                train_overrides={"label_action": label_action},
                action_domain="label",
                action_id=label_action,
                expected_effect=[f"Audit labels before trusting model-side changes: {label_action}."],
                rationale=f"Single-variable test for label action: {label_action}.",
                deployment=deployment,
                priority_hint=1.8,
            )
        )
        changed_variables.setdefault("label_action", []).append(label_action)

    for action in _action_policy_variable_actions(action_policy):
        domain = _action_domain(action)
        if domain in {"data", "label"}:
            continue
        overrides = {f"{domain}_action": action.id, **action.target_variables}
        if "imgsz" in action.target_variables and not _imgsz_change_allowed(action.target_variables["imgsz"], fixed_imgsz):
            guardrails.append(
                f"blocked_imgsz_increase: action={action.id} exceeds fixed baseline imgsz={fixed_imgsz}; "
                "keep input size fixed for fair COCO/YOLO26 comparison."
            )
            continue
        policies.append(
            _policy(
                policy_id=f"next_{domain}_{_slug(action.id)}",
                task_spec=task_spec,
                components=[],
                train_overrides=overrides,
                action_domain=domain,
                action_id=action.id,
                expected_effect=[action.expected_effect or action.description],
                rationale=f"Single-variable test for {domain} action: {action.id}.",
                deployment=deployment,
                priority_hint=1.5,
            )
        )
        changed_variables.setdefault(f"{domain}_action", []).append(action.id)

    if augmentation_policy.actions.enable or augmentation_policy.actions.add:
        variables = dedupe_list([*augmentation_policy.actions.enable, *augmentation_policy.actions.add])
        policies.append(
            _policy(
                policy_id="next_augmentation_policy",
                task_spec=task_spec,
                components=[],
                train_overrides={"augmentation_policy": variables},
                action_domain="augmentation",
                action_id="augmentation_policy",
                expected_effect=augmentation_policy.rationale or ["Measure augmentation policy effect."],
                rationale="Single-variable test for augmentation policy bundle.",
                deployment=deployment,
            )
        )
        changed_variables.setdefault("augmentation_policy", []).extend(variables)

    if postprocess_policy.ids:
        policies.append(
            _policy(
                policy_id="next_postprocess_policy",
                task_spec=task_spec,
                components=[],
                train_overrides={"postprocess": postprocess_policy.ids},
                action_domain="postprocess",
                action_id="postprocess_policy",
                expected_effect=["Measure inference-policy impact before changing network architecture."],
                rationale="Single-variable test for post-processing policy.",
                deployment=deployment,
            )
        )
        changed_variables.setdefault("postprocess", []).extend(postprocess_policy.ids)

    evidence_required = dedupe_list(
        [
            *recipe_plan.evidence_required,
            "precision",
            "recall",
            "latency_ms",
            "model_size_mb",
        ]
    )
    guardrails = dedupe_list(
        [
            *recipe_plan.data_checks,
            *sampling_policy.required_checks,
            *guardrails,
            "record_dataset_version",
            "run_smoke_before_training",
            "keep_single_variable_ablation",
            "do_not_increase_imgsz_for_baseline_comparison",
        ]
    )
    return NextRoundPlan(
        candidate_policies=policies,
        changed_variables={key: dedupe_list(value) for key, value in changed_variables.items()},
        evidence_required=evidence_required,
        guardrails=guardrails,
    )


def _policy(
    policy_id: str,
    task_spec: TaskSpec,
    components: list[str],
    train_overrides: dict[str, object],
    action_domain: str,
    action_id: str,
    expected_effect: list[str],
    rationale: str,
    deployment: DeploymentConstraints | None,
    priority_hint: float = 1.0,
) -> CandidatePolicy:
    constraints = []
    max_latency = task_spec.max_latency_ms or (deployment.max_latency_ms if deployment is not None else None)
    max_size = task_spec.max_model_size_mb or (deployment.max_model_size_mb if deployment is not None else None)
    if max_latency is not None:
        constraints.append(PolicyConstraint(name="max_latency_ms", value=max_latency, hard=True))
    if max_size is not None:
        constraints.append(PolicyConstraint(name="max_model_size_mb", value=max_size, hard=True))
    return CandidatePolicy(
        policy_id=policy_id,
        source="rule_engine",
        action_domain=action_domain,  # type: ignore[arg-type]
        action_id=action_id,
        base_model="yolo11n",
        scale="n",
        framework="ultralytics",
        components=components,
        train_overrides=train_overrides,
        constraints=constraints,
        expected_effect=expected_effect,
        priority_hint=priority_hint,
        risk="medium",
        rationale=rationale,
    )


def _action_policy_variable_actions(action_policy: ErrorActionPlan) -> list[ActionPolicy]:
    """Return action-policy recommendations expressed as non-component variables."""
    actions: list[ActionPolicy] = []
    seen: set[str] = set()
    for recommendation in action_policy.recommendations:
        action = recommendation.action
        if not action.target_variables or action.id in seen:
            continue
        seen.add(action.id)
        actions.append(action)
    return actions


def _label_actions(action_policy: ErrorActionPlan, data_checks: list[str]) -> list[str]:
    """Return label/audit actions from recipes and action policies."""
    actions = [
        check
        for check in data_checks
        if any(token in check for token in ("label", "annotation", "audit", "missing"))
    ]
    if any(
        observation.error_type in {"background_confusion", "hard_negative", "label_noise_induced"}
        for observation in action_policy.observations
    ):
        actions.append("check_missing_labels")
    for action in _action_policy_variable_actions(action_policy):
        if _action_domain(action) == "label":
            actions.append(action.id)
    return dedupe_list(actions)


def _action_domain(action: ActionPolicy) -> str:
    """Infer a first-class action domain from an action policy."""
    keys = set(action.target_variables)
    if action.target_components:
        return "model"
    if keys.intersection({"hard_negative_mining", "background_only_sampling", "data_collection", "class_balancing", "class_pair_sampling"}):
        return "data"
    if keys.intersection({"mosaic"}):
        return "augmentation"
    if keys.intersection({"inference_tiling"}):
        return "postprocess"
    if keys.intersection({"label_audit", "annotation_policy_review"}):
        return "label"
    if keys.intersection({"focal_loss_gamma", "imgsz"}):
        return "training"
    return "training"


def _imgsz_change_allowed(value: object, fixed_imgsz: int | None) -> bool:
    """Return whether an imgsz override is allowed under fixed-baseline rules."""
    if fixed_imgsz is None:
        return True
    try:
        requested = int(value)
    except (TypeError, ValueError):
        return False
    return requested <= fixed_imgsz


def _deployment_answer(task_spec: TaskSpec, deployment: DeploymentConstraints) -> str:
    if task_spec.max_latency_ms or deployment.max_latency_ms:
        return "Latency is a hard planning constraint; high-cost recipes need separate validation."
    if task_spec.max_model_size_mb or deployment.max_model_size_mb:
        return "Model size is a hard planning constraint; prefer data/postprocess checks before larger models."
    return "Deployment constraints are present but not numerically tight."


def _deployment_signals(task_spec: TaskSpec, deployment: DeploymentConstraints) -> list[str]:
    signals: list[str] = [f"deployment_target={deployment.target}", f"preferred_export={deployment.preferred_export}"]
    max_latency = task_spec.max_latency_ms or deployment.max_latency_ms
    max_size = task_spec.max_model_size_mb or deployment.max_model_size_mb
    if max_latency is not None:
        signals.append(f"max_latency_ms={max_latency}")
    if max_size is not None:
        signals.append(f"max_model_size_mb={max_size}")
    return signals


def _default_evidence_status(evidence_required: list[str]) -> dict[str, str]:
    return {evidence: "missing" for evidence in evidence_required}


def _slug(value: str) -> str:
    return value.replace(".", "_").replace("-", "_")
