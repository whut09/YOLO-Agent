"""Runtime-safe shadow and evidence-gated active YOLO26 assignment plugins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.components.assignment import (
    AssignerInputs,
    AssignmentComparison,
    NativeYOLO26AssignerPlugin,
    YOLO26AssignerPlugin,
    build_yolo26_assigner_plugin,
    compare_assignments,
)


AssignmentMethod = Literal[
    "tood_tal",
    "ota",
    "dsla",
    "task_aligned_weighting",
    "dynamic_topk",
    "quality_aware",
    "soft_label",
    "dual_path",
    "conflict_aware",
]
AssignmentMode = Literal["shadow", "active"]
AssignmentScope = Literal["one_to_many", "one_to_one", "both"]


@dataclass(frozen=True)
class _AssignmentSpec:
    component_id: str
    method: AssignmentMethod
    changed_variable: str
    paper_id: str
    adaptation: str
    supported_paths: tuple[Literal["one_to_many", "one_to_one"], ...] = (
        "one_to_many",
    )


ASSIGNMENT_SPECS = {
    item.component_id: item
    for item in (
        _AssignmentSpec(
            component_id="assigner.task_aligned",
            method="tood_tal",
            changed_variable="assignment.one_to_many.tood_tal.mode",
            paper_id="arxiv:2108.07755",
            adaptation=(
                "TOOD Task Alignment Learning only; the task-aligned head is not "
                "included and this is not an exact full-method reproduction."
            ),
        ),
        _AssignmentSpec(
            component_id="assigner.optimal_transport",
            method="ota",
            changed_variable="assignment.one_to_many.ota.mode",
            paper_id="arxiv:2103.14259",
            adaptation=(
                "Dynamic positive supply and entropic Sinkhorn transport over "
                "YOLO26 point candidates; native head and losses are retained."
            ),
        ),
        _AssignmentSpec(
            component_id="assigner.dynamic_smooth_label",
            method="dsla",
            changed_variable="assignment.one_to_many.dsla.mode",
            paper_id="arxiv:2208.00817",
            adaptation=(
                "DSLA interval relaxation, core-zone centerness, and online IoU "
                "adapted to YOLO26 P3-P5 point candidates without a centerness head."
            ),
        ),
        _AssignmentSpec(
            component_id="assigner.task_aligned_weighting",
            method="task_aligned_weighting",
            changed_variable="assignment.one_to_many.task_aligned_weighting.mode",
            paper_id="method-profile:task-aligned-weighting",
            adaptation="Reusable task-aligned weighting over YOLO26 point candidates.",
        ),
        _AssignmentSpec(
            component_id="assigner.dynamic_topk",
            method="dynamic_topk",
            changed_variable="assignment.one_to_many.dynamic_topk.mode",
            paper_id="method-profile:dynamic-topk",
            adaptation="Dynamic positive cardinality over YOLO26 point candidates.",
        ),
        _AssignmentSpec(
            component_id="assigner.quality_aware",
            method="quality_aware",
            changed_variable="assignment.one_to_many.quality_aware.mode",
            paper_id="method-profile:quality-aware-matching",
            adaptation="Classification-IoU quality ranking with native YOLO26 losses.",
        ),
        _AssignmentSpec(
            component_id="assigner.soft_label",
            method="soft_label",
            changed_variable="assignment.one_to_many.soft_label.mode",
            paper_id="method-profile:soft-label-assignment",
            adaptation="Bounded positive-quality smoothing after point matching.",
        ),
        _AssignmentSpec(
            component_id="assigner.dual_path",
            method="dual_path",
            changed_variable="assignment.dual_path.mode",
            paper_id="method-profile:dual-path-assignment",
            adaptation=(
                "Path-specific positive cardinality for native one-to-many and "
                "one-to-one training branches."
            ),
            supported_paths=("one_to_many", "one_to_one"),
        ),
        _AssignmentSpec(
            component_id="assigner.conflict_aware",
            method="conflict_aware",
            changed_variable="assignment.one_to_many.conflict_aware.mode",
            paper_id="method-profile:conflict-aware-positive-selection",
            adaptation="Rejects ambiguous point-to-GT claims using a quality margin.",
        ),
    )
}


class AssignmentPaperPrior(BaseModel):
    paper_id: str
    evidence_level: Literal["paper_prior"] = "paper_prior"
    reported_delta: dict[str, float] = Field(default_factory=dict)
    exact_reproduction: Literal[False] = False
    adaptation: str


class AssignmentRuntimeConfig(BaseModel):
    component_id: str
    method: AssignmentMethod
    changed_variable: str
    assignment_path: AssignmentScope = "one_to_many"
    mode: AssignmentMode = "shadow"
    imgsz: int = 640
    minimum_shadow_batches: int = Field(default=10, ge=1)
    maximum_conflict_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    evidence_interval: int = Field(default=10, ge=1)
    shadow_evidence_path: str | None = None
    shadow_payload_hash: str | None = None
    method_options: dict[str, Any] = Field(default_factory=dict)
    paper_prior: AssignmentPaperPrior

    @model_validator(mode="after")
    def validate_boundary(self) -> "AssignmentRuntimeConfig":
        if self.imgsz != 640:
            raise ValueError("YOLO26 assignment plugins require imgsz=640")
        spec = ASSIGNMENT_SPECS.get(self.component_id)
        if spec is None or spec.method != self.method:
            raise ValueError("assignment component and method do not match")
        if self.changed_variable != spec.changed_variable:
            raise ValueError("assignment changed variable is not canonical")
        requested_paths = set(_paths_for_scope(self.assignment_path))
        if not requested_paths.issubset(spec.supported_paths):
            raise ValueError("assignment mechanism does not support requested path scope")
        if self.method == "dual_path" and self.assignment_path != "both":
            raise ValueError("dual-path assignment requires assignment_path=both")
        if self.mode == "active" and (
            not self.shadow_evidence_path or not self.shadow_payload_hash
        ):
            raise ValueError(
                "active assignment requires prior shadow evidence and payload hash"
            )
        return self


class NativeAssignmentPathAudit(BaseModel):
    path: Literal["one_to_many", "one_to_one"]
    assigner_class: str
    assigner_module: str
    topk: int
    topk2: int
    alpha: float
    beta: float
    strides: list[float]
    use_dfl: bool
    reg_max: int


class YOLO26AssignmentAudit(BaseModel):
    schema_version: str = "yolo26_assignment_audit.v1"
    ultralytics_version: str
    criterion_class: str
    one_to_many: NativeAssignmentPathAudit
    one_to_one: NativeAssignmentPathAudit
    nms_free: bool
    dfl_free: bool
    stal_spatial_behavior_verified: bool
    stal_runtime_form: str
    native_loss_inputs: list[str]
    native_loss_outputs: list[str]
    verified: bool


class AssignmentEvidenceAggregate(BaseModel):
    batches: int = 0
    total_candidates: int = 0
    baseline_positive_count: int = 0
    candidate_positive_count: int = 0
    foreground_disagreement_count: int = 0
    gt_conflict_count: int = 0
    conflict_count: int = 0
    baseline_positive_ratio: float = 0.0
    candidate_positive_ratio: float = 0.0
    conflict_rate: float = 0.0
    gt_conflict_rate: float = 0.0
    matching_stability: float = 0.0

    def add(self, comparison: AssignmentComparison) -> None:
        self.batches += 1
        self.total_candidates += comparison.total_candidates
        self.baseline_positive_count += comparison.baseline_positive_count
        self.candidate_positive_count += comparison.candidate_positive_count
        self.foreground_disagreement_count += comparison.foreground_disagreement_count
        self.gt_conflict_count += comparison.gt_conflict_count
        self.conflict_count += comparison.conflict_count
        denominator = max(self.total_candidates, 1)
        self.baseline_positive_ratio = self.baseline_positive_count / denominator
        self.candidate_positive_ratio = self.candidate_positive_count / denominator
        self.conflict_rate = self.conflict_count / denominator
        self.gt_conflict_rate = self.gt_conflict_count / denominator
        self.matching_stability = 1.0 - self.conflict_rate


class AssignmentShadowEvidence(BaseModel):
    schema_version: str = "yolo26_assignment_shadow_evidence.v1"
    component_id: str
    method: AssignmentMethod
    assignment_path: AssignmentScope
    mode: AssignmentMode
    protocol_hash: str
    runtime_payload_hash: str
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    adapter_version: str
    runtime_plugin_version: str
    runtime_plugin_sha256: str
    rank: int
    native_baseline_plugin: str
    candidate_plugin: str
    candidate_plugin_version: str
    anchor_representation: Literal["point"] = "point"
    replaces_head: Literal[False] = False
    replaces_loss: Literal[False] = False
    changes_inference_path: Literal[False] = False
    assignment_path_replaced: str | None = None
    assignment_paths_replaced: list[
        Literal["one_to_many", "one_to_one"]
    ] = Field(default_factory=list)
    native_audit: YOLO26AssignmentAudit
    aggregate: AssignmentEvidenceAggregate = Field(default_factory=AssignmentEvidenceAggregate)
    path_aggregates: dict[
        Literal["one_to_many", "one_to_one"], AssignmentEvidenceAggregate
    ] = Field(default_factory=dict)
    output_validation_failures: list[str] = Field(default_factory=list)
    shadow_passed: bool = False
    activation_source_evidence: str | None = None
    activation_source_sha256: str | None = None
    paper_prior: AssignmentPaperPrior
    checkpoint_metadata_paths: list[str] = Field(default_factory=list)


class AssignmentActivationDecision(BaseModel):
    allowed: bool
    blocked_by: list[str] = Field(default_factory=list)
    evidence_path: str | None = None
    evidence_sha256: str | None = None
    runtime_payload_hash: str | None = None


class AssignmentActivationGate:
    """Require valid shadow evidence before replacing one training path."""

    def evaluate(
        self,
        evidence_path: Path | str,
        *,
        component_id: str,
        method: AssignmentMethod,
        assignment_path: AssignmentScope,
        minimum_batches: int,
        maximum_conflict_rate: float,
        protocol_hash: str | None = None,
        shadow_payload_hash: str | None = None,
        runtime_plugin_sha256: str | None = None,
        changed_variable: str | None = None,
    ) -> AssignmentActivationDecision:
        path = Path(evidence_path)
        blocked: list[str] = []
        if not path.is_file():
            return AssignmentActivationDecision(
                allowed=False,
                blocked_by=["shadow_evidence_missing"],
                evidence_path=str(path),
            )
        try:
            evidence = AssignmentShadowEvidence.model_validate_json(
                path.read_text(encoding="utf-8-sig")
            )
        except (OSError, ValueError) as exc:
            return AssignmentActivationDecision(
                allowed=False,
                blocked_by=[f"shadow_evidence_invalid:{exc}"],
                evidence_path=str(path),
            )
        if evidence.component_id != component_id or evidence.method != method:
            blocked.append("shadow_evidence_component_mismatch")
        if protocol_hash is not None and evidence.protocol_hash != protocol_hash:
            blocked.append("shadow_evidence_protocol_mismatch")
        if (
            shadow_payload_hash is not None
            and evidence.runtime_payload_hash != shadow_payload_hash
        ):
            blocked.append("shadow_evidence_payload_mismatch")
        if (
            runtime_plugin_sha256 is not None
            and evidence.runtime_plugin_sha256 != runtime_plugin_sha256
        ):
            blocked.append("shadow_evidence_plugin_mismatch")
        if changed_variable is not None and evidence.changed_variables != {
            changed_variable: "shadow"
        }:
            blocked.append("shadow_evidence_changed_variable_mismatch")
        if evidence.assignment_path != assignment_path or evidence.mode != "shadow":
            blocked.append("shadow_evidence_path_or_mode_mismatch")
        required_paths = _paths_for_scope(assignment_path)
        for required_path in required_paths:
            path_aggregate = evidence.path_aggregates.get(required_path)
            if path_aggregate is None:
                blocked.append(f"shadow_evidence_path_missing:{required_path}")
                continue
            if path_aggregate.batches < minimum_batches:
                blocked.append(
                    f"shadow_evidence_path_batches_insufficient:{required_path}:"
                    f"{path_aggregate.batches}/{minimum_batches}"
                )
            if path_aggregate.baseline_positive_count == 0:
                blocked.append(f"shadow_path_baseline_has_no_positives:{required_path}")
            if path_aggregate.candidate_positive_count == 0:
                blocked.append(f"shadow_path_candidate_has_no_positives:{required_path}")
            if path_aggregate.conflict_rate > maximum_conflict_rate:
                blocked.append(
                    f"shadow_path_conflict_rate_exceeded:{required_path}:"
                    f"{path_aggregate.conflict_rate:.6f}>"
                    f"{maximum_conflict_rate:.6f}"
                )
        if evidence.aggregate.batches < minimum_batches:
            blocked.append(
                f"shadow_evidence_batches_insufficient:"
                f"{evidence.aggregate.batches}/{minimum_batches}"
            )
        if evidence.aggregate.baseline_positive_count == 0:
            blocked.append("shadow_baseline_has_no_positives")
        if evidence.aggregate.candidate_positive_count == 0:
            blocked.append("shadow_candidate_has_no_positives")
        if evidence.aggregate.conflict_rate > maximum_conflict_rate:
            blocked.append(
                f"shadow_conflict_rate_exceeded:"
                f"{evidence.aggregate.conflict_rate:.6f}>{maximum_conflict_rate:.6f}"
            )
        if evidence.output_validation_failures:
            blocked.append("shadow_output_validation_failed")
        if not evidence.native_audit.verified:
            blocked.append("native_assignment_audit_failed")
        if not evidence.shadow_passed:
            blocked.append("shadow_evidence_not_passed")
        return AssignmentActivationDecision(
            allowed=not blocked,
            blocked_by=blocked,
            evidence_path=str(path.resolve()),
            evidence_sha256=_sha256(path),
            runtime_payload_hash=evidence.runtime_payload_hash,
        )


class YOLO26AssignmentRuntimePlugin:
    """Observe or evidence-gated replace exactly one YOLO26 assignment path."""

    plugin_version = "yolo26_assignment_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = AssignmentRuntimeConfig.model_validate(options)
        self.candidate = build_yolo26_assigner_plugin(
            self.config.method,
            **self.config.method_options,
        )
        self.evidence: AssignmentShadowEvidence | None = None
        self._active_wrappers: dict[
            Literal["one_to_many", "one_to_one"], _ActiveAssignerWrapper
        ] = {}
        self._activation_decision: AssignmentActivationDecision | None = None

    def prepare_command(
        self,
        *,
        payload: AdapterRuntimePayload,
        command: list[str],
        env: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        del payload
        filtered = [
            token
            for token in command
            if not token.partition("=")[0].startswith("assignment.")
        ]
        return filtered, env

    def build_criterion(
        self,
        *,
        context: Any,
        trainer: Any,
        model: Any,
        criterion: Any,
    ) -> Any:
        del trainer
        audit = audit_yolo26_assignment_runtime(model, criterion)
        self._ensure_evidence(context, audit)
        if self.config.mode == "active":
            decision = AssignmentActivationGate().evaluate(
                self.config.shadow_evidence_path or "",
                component_id=self.config.component_id,
                method=self.config.method,
                assignment_path=self.config.assignment_path,
                minimum_batches=self.config.minimum_shadow_batches,
                maximum_conflict_rate=self.config.maximum_conflict_rate,
                protocol_hash=context.payload.protocol_hash,
                shadow_payload_hash=self.config.shadow_payload_hash,
                runtime_plugin_sha256=_sha256(Path(__file__)),
                changed_variable=self.config.changed_variable,
            )
            self._activation_decision = decision
            if not decision.allowed:
                raise ValueError(
                    "active assignment blocked: " + "; ".join(decision.blocked_by)
                )
            for path in _paths_for_scope(self.config.assignment_path):
                native = _criterion_path(criterion, path)
                wrapper = _ActiveAssignerWrapper(
                    plugin=self.candidate,
                    path=path,
                    num_classes=native.nc,
                    strides=[float(value) for value in native.stride.tolist()],
                )
                self._active_wrappers[path] = wrapper
                native.assigner = wrapper
            if self.evidence is not None:
                self.evidence.assignment_path_replaced = self.config.assignment_path
                self.evidence.assignment_paths_replaced = list(
                    _paths_for_scope(self.config.assignment_path)
                )
                self.evidence.activation_source_evidence = decision.evidence_path
                self.evidence.activation_source_sha256 = decision.evidence_sha256
                self._persist(context)
        return criterion

    def compute_loss(
        self,
        *,
        context: Any,
        trainer: Any,
        model: Any,
        criterion: Any,
        predictions: Any,
        batch: dict[str, Any],
        loss_output: Any,
    ) -> Any:
        del trainer, model
        if self.evidence is None:
            raise RuntimeError("assignment runtime evidence was not initialized")
        if self.config.mode == "shadow":
            try:
                for path in _paths_for_scope(self.config.assignment_path):
                    inputs = extract_assignment_inputs(
                        criterion,
                        predictions,
                        batch,
                        path=path,
                    )
                    native = _criterion_path(criterion, path)
                    baseline = NativeYOLO26AssignerPlugin(native.assigner).run(inputs)
                    candidate = self.candidate.run(inputs)
                    comparison = compare_assignments(baseline, candidate)
                    self.evidence.aggregate.add(comparison)
                    path_aggregate = self.evidence.path_aggregates.setdefault(
                        path,
                        AssignmentEvidenceAggregate(),
                    )
                    path_aggregate.add(comparison)
            except (RuntimeError, TypeError, ValueError) as exc:
                self.evidence.output_validation_failures.append(str(exc))
                self._persist(context)
                raise
            self._update_shadow_passed()
        else:
            missing_paths = [
                path
                for path in _paths_for_scope(self.config.assignment_path)
                if path not in self._active_wrappers
                or self._active_wrappers[path].calls == 0
            ]
            if missing_paths:
                raise RuntimeError(
                    "active assignment paths did not execute: "
                    + ", ".join(missing_paths)
                )
        if (
            self.evidence.aggregate.batches <= 1
            or self.evidence.aggregate.batches % self.config.evidence_interval == 0
        ):
            self._persist(context)
        # Shadow mode is observational: native loss tensors are returned unchanged.
        return loss_output

    def on_checkpoint_save(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoints: dict[str, Any],
    ) -> None:
        del trainer
        if self.evidence is None:
            return
        for checkpoint in checkpoints.values():
            path = Path(checkpoint) if checkpoint else None
            if path is None or not path.is_file():
                continue
            metadata_path = path.with_suffix(
                path.suffix + f".assignment.{self.config.method}.{self.config.mode}.json"
            )
            payload = {
                **self.evidence.model_dump(mode="json"),
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": _sha256(path),
            }
            _write_json_atomic(metadata_path, payload)
            resolved = str(metadata_path.resolve())
            if resolved not in self.evidence.checkpoint_metadata_paths:
                self.evidence.checkpoint_metadata_paths.append(resolved)
        self._persist(context)

    def _ensure_evidence(
        self,
        context: Any,
        audit: YOLO26AssignmentAudit,
    ) -> AssignmentShadowEvidence:
        if self.evidence is None:
            self.evidence = AssignmentShadowEvidence(
                component_id=self.config.component_id,
                method=self.config.method,
                assignment_path=self.config.assignment_path,
                mode=self.config.mode,
                protocol_hash=context.payload.protocol_hash,
                runtime_payload_hash=str(getattr(context.payload, "payload_hash", "")),
                changed_variables=dict(getattr(context.payload, "changed_variables", {})),
                adapter_version=YOLO26AssignmentAdapter.adapter_version,
                runtime_plugin_version=self.plugin_version,
                runtime_plugin_sha256=_sha256(Path(__file__)),
                rank=_rank(),
                native_baseline_plugin=NativeYOLO26AssignerPlugin.plugin_id,
                candidate_plugin=self.candidate.plugin_id,
                candidate_plugin_version=self.candidate.plugin_version,
                native_audit=audit,
                paper_prior=self.config.paper_prior,
            )
            self._persist(context)
        return self.evidence

    def _update_shadow_passed(self) -> None:
        if self.evidence is None:
            return
        aggregate = self.evidence.aggregate
        path_aggregates = self.evidence.path_aggregates
        required_paths = _paths_for_scope(self.config.assignment_path)
        self.evidence.shadow_passed = bool(
            path_aggregates
            and all(
                path in path_aggregates
                and path_aggregates[path].batches
                >= self.config.minimum_shadow_batches
                and path_aggregates[path].baseline_positive_count > 0
                and path_aggregates[path].candidate_positive_count > 0
                for path in required_paths
            )
            and aggregate.baseline_positive_count > 0
            and aggregate.candidate_positive_count > 0
            and aggregate.conflict_rate <= self.config.maximum_conflict_rate
            and not self.evidence.output_validation_failures
            and self.evidence.native_audit.verified
        )

    def _persist(self, context: Any) -> None:
        if self.evidence is not None:
            _write_json_atomic(
                _evidence_path(
                    context.payload_path.parent,
                    self.config.method,
                    self.config.mode,
                ),
                self.evidence.model_dump(mode="json"),
            )


class YOLO26AssignmentAdapter(ComponentAdapter):
    """Materialize independent assignment methods through the runtime bridge."""

    adapter_version = "yolo26_assignment_adapter.v1"
    source_commit = "yolo-agent:yolo26-assignment-plugin-v1"
    strategy = "assigner_injection"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset(
        spec.changed_variable for spec in ASSIGNMENT_SPECS.values()
    )

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            import torch
            import ultralytics

            return AdapterValidationReport(
                ok=True,
                checks={
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                },
            )
        except ImportError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        errors: list[str] = []
        if context.contract.component_id not in ASSIGNMENT_SPECS:
            errors.append("unknown YOLO26 assignment component")
        if context.detector_family != "yolo26":
            errors.append("assignment plugins support YOLO26 only")
        if context.imgsz != 640:
            errors.append("assignment plugins require fixed imgsz=640")
        if bool(context.options.get("anchor_based")) or bool(
            context.options.get("requires_anchors")
        ):
            errors.append("anchor-based assignment cannot be attached to YOLO26")
        path = str(context.options.get("assignment_path", "one_to_many"))
        spec = ASSIGNMENT_SPECS.get(context.contract.component_id)
        if spec is not None:
            try:
                requested_paths = set(_paths_for_scope(path))
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if not requested_paths.issubset(spec.supported_paths):
                    errors.append("assignment path scope is unsupported by component")
                if spec.method == "dual_path" and path != "both":
                    errors.append("dual-path assignment must declare both paths")
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "declared_path": path,
                "anchor_representation": "point",
                "one_to_one_preserved": path != "both",
                "nms_free_preserved": True,
                "dfl_free_preserved": True,
                "head_preserved": True,
                "native_loss_preserved_in_shadow": True,
            },
        )

    def patch_model_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        return config

    def patch_training_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        runtime = _runtime_config(context)
        config[runtime.changed_variable] = runtime.mode
        return config

    def build_module(self, context: AdapterContext) -> YOLO26AssignerPlugin:
        runtime = _runtime_config(context)
        return build_yolo26_assigner_plugin(runtime.method, **runtime.method_options)

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        return WeightLoadResult(
            loaded=False,
            message="assignment plugins have no trainable adapter weights",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            runtime = _runtime_config(context)
            plugin = build_yolo26_assigner_plugin(
                runtime.method,
                **runtime.method_options,
            )
            inputs = _synthetic_inputs()
            output = plugin.run(inputs)
            return SmokeTestResult(
                passed=bool(output.foreground_mask.any()),
                evidence_kind="local",
                checks={
                    "shape": str(tuple(output.target_scores.shape)),
                    "positive_count": str(int(output.foreground_mask.sum().item())),
                    "point_based": True,
                    "assignment_path": runtime.assignment_path,
                    "shadow_mode": runtime.mode == "shadow",
                    "native_loss_unchanged": runtime.mode == "shadow",
                    "imgsz": "640",
                },
            )
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[str(exc)],
            )

    def gpu_smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is None or not torch.cuda.is_available():
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                checks={"gpu_smoke_implemented": True, "cuda_available": False},
                errors=["cuda_not_available"],
            )
        try:
            checks = _run_gpu_shadow_smoke(self, context)
            return SmokeTestResult(
                passed=all(value is True for value in checks.values()),
                evidence_kind="local",
                checks=checks,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                checks={"gpu_smoke_implemented": True, "cuda_available": True},
                errors=[str(exc)],
            )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        runtime = _runtime_config(context)
        return [
            ExpectedArtifact(
                name=f"assignment_{runtime.method}_{runtime.mode}_evidence",
                relative_path=Path(
                    f"assignment_{runtime.method}_{runtime.mode}_evidence.json"
                ),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        runtime = _runtime_config(context)
        return RollbackPlan(
            actions=[
                f"remove {runtime.method} {runtime.mode} assignment plugin and sidecars"
            ],
            files_to_remove=[
                Path(f"assignment_{runtime.method}_{runtime.mode}_evidence.json")
            ],
        )

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        runtime = _runtime_config(context)
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            assigner_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.assigners.yolo26_assignment:"
                        "YOLO26AssignmentRuntimePlugin"
                    ),
                    options=runtime.model_dump(mode="json"),
                    required_hooks=["compute_loss"],
                )
            ],
            generated_config=generated_config,
            changed_variables={runtime.changed_variable: runtime.mode},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )


class _ActiveAssignerWrapper:
    """Adapt the typed plugin back to the exact native callable signature."""

    def __init__(
        self,
        *,
        plugin: YOLO26AssignerPlugin,
        path: Literal["one_to_many", "one_to_one"],
        num_classes: int,
        strides: list[float],
    ) -> None:
        self.plugin = plugin
        self.path = path
        self.num_classes = num_classes
        self.strides = strides
        self.calls = 0
        self.last_positive_count = 0

    def __call__(
        self,
        predicted_scores: Any,
        predicted_boxes: Any,
        anchor_points: Any,
        gt_labels: Any,
        gt_boxes: Any,
        gt_mask: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        strides = _infer_point_strides(anchor_points, self.strides)
        output = self.plugin.run(
            AssignerInputs(
                predicted_scores=predicted_scores,
                predicted_boxes_xyxy=predicted_boxes,
                anchor_points_xy=anchor_points,
                stride_per_anchor=strides,
                gt_labels=gt_labels,
                gt_boxes_xyxy=gt_boxes,
                gt_mask=gt_mask,
                num_classes=self.num_classes,
                path=self.path,
            )
        )
        self.calls += 1
        self.last_positive_count = int(output.foreground_mask.sum().item())
        return output.native_tuple()


def audit_yolo26_assignment_runtime(model: Any, criterion: Any) -> YOLO26AssignmentAudit:
    """Audit the installed E2E assignment and loss contract before any hook."""
    import ultralytics

    one_to_many = _criterion_path(criterion, "one_to_many")
    one_to_one = _criterion_path(criterion, "one_to_one")
    audits = {
        "one_to_many": _audit_path("one_to_many", one_to_many),
        "one_to_one": _audit_path("one_to_one", one_to_one),
    }
    assigner_source = inspect.getsource(type(one_to_many.assigner).select_candidates_in_gts)
    spatial = all(
        token in assigner_source
        for token in ("wh_mask", "stride_val", "smallest stride")
    )
    head = getattr(model, "model", [None])[-1]
    nms_free = bool(
        getattr(model, "end2end", False) or getattr(head, "end2end", False)
    )
    dfl_free = not one_to_many.use_dfl and not one_to_one.use_dfl
    verified = bool(
        type(criterion).__name__ == "E2ELoss"
        and audits["one_to_many"].topk == 10
        and audits["one_to_one"].topk == 7
        and audits["one_to_one"].topk2 == 1
        and spatial
        and nms_free
        and dfl_free
    )
    return YOLO26AssignmentAudit(
        ultralytics_version=ultralytics.__version__,
        criterion_class=type(criterion).__name__,
        one_to_many=audits["one_to_many"],
        one_to_one=audits["one_to_one"],
        nms_free=nms_free,
        dfl_free=dfl_free,
        stal_spatial_behavior_verified=spatial,
        stal_runtime_form=(
            "TaskAlignedAssigner small-box spatial expansion; no distinct STAL class"
        ),
        native_loss_inputs=[
            "one2many/one2one boxes",
            "scores",
            "feature maps",
            "batch indices/classes/xywh boxes",
        ],
        native_loss_outputs=["box_loss", "cls_loss", "dfl_slot_zero_when_reg_max_1"],
        verified=verified,
    )


def extract_assignment_inputs(
    criterion: Any,
    predictions: Any,
    batch: dict[str, Any],
    *,
    path: Literal["one_to_many", "one_to_one"],
) -> AssignerInputs:
    """Decode the selected native branch without changing criterion state."""
    import torch
    from ultralytics.utils.tal import make_anchors

    native = _criterion_path(criterion, path)
    branches = _prediction_branches(predictions)
    branch = branches[path]
    scores = branch["scores"].permute(0, 2, 1).contiguous().detach().sigmoid()
    distribution = branch["boxes"].permute(0, 2, 1).contiguous().detach()
    anchor_points, stride_tensor = make_anchors(branch["feats"], native.stride, 0.5)
    batch_size = scores.shape[0]
    image_size = (
        torch.tensor(
            branch["feats"][0].shape[2:],
            device=native.device,
            dtype=scores.dtype,
        )
        * native.stride[0]
    )
    targets = torch.cat(
        (
            batch["batch_idx"].view(-1, 1),
            batch["cls"].view(-1, 1),
            batch["bboxes"],
        ),
        dim=1,
    )
    with torch.no_grad():
        targets = native.preprocess(
            targets.to(native.device),
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        gt_labels, gt_boxes = targets.split((1, 4), dim=2)
        gt_mask = gt_boxes.sum(2, keepdim=True).gt_(0.0)
        predicted_boxes = native.bbox_decode(anchor_points, distribution) * stride_tensor
    return AssignerInputs(
        predicted_scores=scores,
        predicted_boxes_xyxy=predicted_boxes,
        anchor_points_xy=anchor_points * stride_tensor,
        stride_per_anchor=stride_tensor,
        gt_labels=gt_labels,
        gt_boxes_xyxy=gt_boxes,
        gt_mask=gt_mask,
        num_classes=native.nc,
        path=path,
    )


def _runtime_config(context: AdapterContext) -> AssignmentRuntimeConfig:
    spec = _spec(context)
    mode = str(context.options.get(spec.changed_variable, "shadow"))
    shadow_path = context.options.get("assignment.shadow_evidence_path")
    shadow_payload_hash = context.options.get("assignment.shadow_payload_hash")
    method_options = context.options.get(f"assignment.{spec.method}.options", {})
    return AssignmentRuntimeConfig(
        component_id=spec.component_id,
        method=spec.method,
        changed_variable=spec.changed_variable,
        assignment_path=str(
            context.options.get(
                "assignment_path",
                "both" if spec.method == "dual_path" else "one_to_many",
            )
        ),
        mode=mode,
        imgsz=context.imgsz,
        minimum_shadow_batches=int(
            context.options.get("assignment.minimum_shadow_batches", 10)
        ),
        maximum_conflict_rate=float(
            context.options.get("assignment.maximum_conflict_rate", 0.95)
        ),
        evidence_interval=int(context.options.get("assignment.evidence_interval", 10)),
        shadow_evidence_path=str(shadow_path) if shadow_path else None,
        shadow_payload_hash=(
            str(shadow_payload_hash) if shadow_payload_hash else None
        ),
        method_options=dict(method_options) if isinstance(method_options, dict) else {},
        paper_prior=AssignmentPaperPrior(
            paper_id=spec.paper_id,
            adaptation=spec.adaptation,
        ),
    )


def _run_gpu_shadow_smoke(
    adapter: YOLO26AssignmentAdapter,
    context: AdapterContext,
) -> dict[str, bool]:
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    from yolo_agent.adapters.ultralytics.plugin_bridge import (
        PluginCriterionWrapper,
        UltralyticsTrainerPluginBridge,
    )

    workspace = context.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    options = dict(context.options)
    options.update(
        {
            "assignment.minimum_shadow_batches": 1,
            "assignment.maximum_conflict_rate": 1.0,
            "assignment.evidence_interval": 1,
        }
    )
    gpu_context = context.model_copy(update={"workspace": workspace, "options": options})
    runtime = _runtime_config(gpu_context)
    if runtime.mode != "shadow":
        raise ValueError("assignment GPU certification requires shadow mode")
    payload = adapter.build_runtime_payload(
        gpu_context,
        protocol_hash="assignment-gpu-smoke-imgsz-640",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    payload_path = payload.write(workspace / "assignment_gpu_runtime.yaml")
    bridge = UltralyticsTrainerPluginBridge(payload_path)
    model = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False).cuda()
    model.args = get_cfg(overrides={"imgsz": 640})
    model.train()
    trainer = type("Trainer", (), {"args": get_cfg(overrides={"imgsz": 640})})()
    bridge.install_model_hooks(model, trainer=trainer)
    wrapped = model.init_criterion()
    if not isinstance(wrapped, PluginCriterionWrapper):
        raise TypeError("assignment GPU smoke did not install criterion wrapper")
    native_criterion = wrapped.criterion
    native_one_to_many = native_criterion.one2many.assigner
    native_one_to_one = native_criterion.one2one.assigner
    image = torch.rand(1, 3, 64, 64, device="cuda")
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0], device="cuda"),
        "cls": torch.tensor([[0.0]], device="cuda"),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]], device="cuda"),
    }
    predictions = model(image)
    native_loss, native_items = native_criterion(predictions, batch)
    shadow_loss, shadow_items = wrapped(predictions, batch)
    native_equivalent = bool(
        torch.equal(shadow_loss, native_loss)
        and torch.equal(shadow_items, native_items)
    )
    shadow_loss.sum().backward()
    backward = any(parameter.grad is not None for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_predictions = model(image)
        amp_loss, _ = wrapped(amp_predictions, batch)
    amp_loss.sum().backward()
    evidence_path = _evidence_path(workspace, runtime.method, "shadow")
    evidence = AssignmentShadowEvidence.model_validate_json(
        evidence_path.read_text(encoding="utf-8-sig")
    )
    aggregate = evidence.aggregate
    return {
        "gpu_smoke_implemented": True,
        "cuda_available": True,
        "shadow_mode_only": evidence.mode == "shadow",
        "native_loss_equivalent": native_equivalent,
        "native_one_to_one_preserved": bool(
            native_criterion.one2many.assigner is native_one_to_many
            and native_criterion.one2one.assigner is native_one_to_one
        ),
        "native_audit_verified": evidence.native_audit.verified,
        "positive_ratio_recorded": bool(
            aggregate.baseline_positive_count > 0
            and aggregate.candidate_positive_count > 0
        ),
        "conflict_rate_recorded": 0.0 <= aggregate.conflict_rate <= 1.0,
        "shadow_artifact_passed": evidence.shadow_passed,
        "backward": backward,
        "amp": bool(torch.isfinite(amp_loss).all()),
        "fixed_imgsz_640": runtime.imgsz == 640,
    }


def _spec(context: AdapterContext) -> _AssignmentSpec:
    try:
        return ASSIGNMENT_SPECS[context.contract.component_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported YOLO26 assignment component: {context.contract.component_id}"
        ) from exc


def _criterion_path(criterion: Any, path: str) -> Any:
    attribute = {
        "one_to_many": "one2many",
        "one_to_one": "one2one",
    }.get(path, path)
    native = getattr(criterion, attribute, None)
    if native is None:
        raise ValueError(f"YOLO26 E2ELoss is missing assignment path {path}")
    required = (
        "assigner",
        "bbox_decode",
        "preprocess",
        "stride",
        "device",
        "nc",
        "reg_max",
        "use_dfl",
    )
    for name in required:
        if not hasattr(native, name):
            raise ValueError(f"YOLO26 {path} criterion is missing {name}")
    return native


def _audit_path(path: Literal["one_to_many", "one_to_one"], native: Any) -> NativeAssignmentPathAudit:
    assigner = native.assigner
    return NativeAssignmentPathAudit(
        path=path,
        assigner_class=type(assigner).__name__,
        assigner_module=type(assigner).__module__,
        topk=int(assigner.topk),
        topk2=int(assigner.topk2),
        alpha=float(assigner.alpha),
        beta=float(assigner.beta),
        strides=[float(value) for value in assigner.stride],
        use_dfl=bool(native.use_dfl),
        reg_max=int(native.reg_max),
    )


def _prediction_branches(predictions: Any) -> dict[str, Any]:
    if isinstance(predictions, tuple):
        predictions = predictions[1]
    if not isinstance(predictions, dict):
        raise ValueError("YOLO26 predictions must contain end-to-end branches")
    required = {"one2many", "one2one"}
    if not required.issubset(predictions):
        raise ValueError("YOLO26 predictions are missing one2many or one2one")
    for path in required:
        branch = predictions[path]
        if not isinstance(branch, dict) or not {"boxes", "scores", "feats"}.issubset(
            branch
        ):
            raise ValueError(f"YOLO26 {path} prediction branch is incomplete")
    return {"one_to_many": predictions["one2many"], "one_to_one": predictions["one2one"]}


def _infer_point_strides(anchor_points: Any, strides: list[float]) -> Any:
    import torch

    result = anchor_points.new_zeros((anchor_points.shape[0], 1))
    assigned = torch.zeros(anchor_points.shape[0], dtype=torch.bool, device=anchor_points.device)
    for stride in sorted(strides):
        remainder = torch.remainder(anchor_points, stride)
        matches = torch.isclose(
            remainder,
            remainder.new_full(remainder.shape, stride / 2.0),
            atol=1e-4,
            rtol=0.0,
        ).all(dim=-1)
        result[matches & ~assigned] = stride
        assigned |= matches
    if not bool(assigned.all()):
        raise ValueError("could not infer YOLO26 point stride for active assignment")
    return result


def _synthetic_inputs() -> AssignerInputs:
    import torch

    points = torch.stack(
        [
            torch.linspace(4.0, 156.0, 20),
            torch.linspace(4.0, 156.0, 20),
        ],
        dim=-1,
    )
    boxes = torch.stack(
        [points[:, 0] - 10, points[:, 1] - 10, points[:, 0] + 10, points[:, 1] + 10],
        dim=-1,
    ).unsqueeze(0)
    scores = torch.full((1, 20, 2), 0.1)
    scores[0, :10, 0] = torch.linspace(0.95, 0.55, 10)
    return AssignerInputs(
        predicted_scores=scores,
        predicted_boxes_xyxy=boxes,
        anchor_points_xy=points,
        stride_per_anchor=torch.full((20, 1), 8.0),
        gt_labels=torch.tensor([[[0.0]]]),
        gt_boxes_xyxy=torch.tensor([[[0.0, 0.0, 96.0, 96.0]]]),
        gt_mask=torch.tensor([[[True]]]),
        num_classes=2,
        path="one_to_many",
    )


def _evidence_path(
    directory: Path,
    method: AssignmentMethod,
    mode: AssignmentMode,
) -> Path:
    rank = _rank()
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    return directory / f"assignment_{method}_{mode}_evidence{suffix}.json"


def _paths_for_scope(
    scope: str,
) -> tuple[Literal["one_to_many", "one_to_one"], ...]:
    if scope == "both":
        return ("one_to_many", "one_to_one")
    if scope == "one_to_many":
        return ("one_to_many",)
    if scope == "one_to_one":
        return ("one_to_one",)
    raise ValueError(f"unsupported assignment path scope: {scope}")


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ASSIGNMENT_SPECS",
    "AssignmentActivationDecision",
    "AssignmentActivationGate",
    "AssignmentEvidenceAggregate",
    "AssignmentRuntimeConfig",
    "AssignmentShadowEvidence",
    "YOLO26AssignmentAdapter",
    "YOLO26AssignmentAudit",
    "YOLO26AssignmentRuntimePlugin",
    "audit_yolo26_assignment_runtime",
    "extract_assignment_inputs",
]
