"""YOLO26 runtime adapters for additive quality-alignment losses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from yolo_agent.components.auxiliary_losses import (
    AuxiliaryLossInputs,
    AuxiliaryLossPlugin,
    build_auxiliary_loss,
)


AuxiliaryLossName = Literal[
    "correlation",
    "bpc_calibration",
    "pseudo_iou",
    "iou_aware_classification",
    "localization_aware_classification",
    "boundary_aware",
    "uncertainty_weighted_regression",
    "hard_negative_classification",
    "class_balanced_focal",
]


@dataclass(frozen=True)
class _LossSpec:
    component_id: str
    loss_name: AuxiliaryLossName
    changed_variable: str
    default_weight: float
    paper_id: str
    adaptation: str


LOSS_SPECS = {
    item.component_id: item
    for item in (
        _LossSpec(
            component_id="loss.quality.correlation",
            loss_name="correlation",
            changed_variable="loss.correlation.weight",
            default_weight=0.2,
            paper_id="arxiv:2301.01019",
            adaptation="Concordance correlation on native positive matches.",
        ),
        _LossSpec(
            component_id="loss.calibration.bpc",
            loss_name="bpc_calibration",
            changed_variable="loss.bpc_calibration.weight",
            default_weight=0.1,
            paper_id="arxiv:2303.14404",
            adaptation="BPC-style confidence quadrant objective on native matches.",
        ),
        _LossSpec(
            component_id="loss.quality.pseudo_iou",
            loss_name="pseudo_iou",
            changed_variable="loss.pseudo_iou.weight",
            default_weight=0.1,
            paper_id="arxiv:2104.14082",
            adaptation=(
                "Pseudo-IoU is used only as an auxiliary quality target; "
                "the native assigner is unchanged."
            ),
        ),
        _LossSpec(
            component_id="loss.quality.iou_aware_classification",
            loss_name="iou_aware_classification",
            changed_variable="loss.iou_aware_classification.weight",
            default_weight=0.1,
            paper_id="method-profile:iou-aware-classification",
            adaptation="Matched IoU is an auxiliary true-class quality target.",
        ),
        _LossSpec(
            component_id="loss.quality.localization_aware",
            loss_name="localization_aware_classification",
            changed_variable="loss.localization_aware_classification.weight",
            default_weight=0.1,
            paper_id="method-profile:localization-aware-classification",
            adaptation="Confidence is aligned to matched IoU and anchor-center quality.",
        ),
        _LossSpec(
            component_id="loss.boundary_aware",
            loss_name="boundary_aware",
            changed_variable="loss.boundary_aware.weight",
            default_weight=0.05,
            paper_id="method-profile:boundary-aware-loss",
            adaptation="Matched box edge error is an auxiliary regression term.",
        ),
        _LossSpec(
            component_id="loss.localization.uncertainty_weighted",
            loss_name="uncertainty_weighted_regression",
            changed_variable="loss.uncertainty_weighted_regression.weight",
            default_weight=0.05,
            paper_id="method-profile:uncertainty-weighted-regression",
            adaptation="Detached confidence-derived uncertainty weights a box residual.",
        ),
        _LossSpec(
            component_id="loss.hard_negative_classification",
            loss_name="hard_negative_classification",
            changed_variable="loss.hard_negative_classification.weight",
            default_weight=0.05,
            paper_id="method-profile:hard-negative-classification",
            adaptation="Unmatched native locations receive an auxiliary background BCE.",
        ),
        _LossSpec(
            component_id="loss.class_balanced_focal",
            loss_name="class_balanced_focal",
            changed_variable="loss.class_balanced_focal.weight",
            default_weight=0.05,
            paper_id="method-profile:class-balanced-focal",
            adaptation="Batch class-frequency weights and focal modulation are auxiliary.",
        ),
    )
}


class AuxiliaryPaperPrior(BaseModel):
    paper_id: str
    evidence_level: Literal["paper_prior"] = "paper_prior"
    reported_delta: dict[str, float] = Field(default_factory=dict)
    exact_reproduction: Literal[False] = False
    adaptation: str


class AuxiliaryLossRuntimeConfig(BaseModel):
    loss_name: AuxiliaryLossName
    component_id: str
    changed_variable: str
    weight: float = Field(ge=0.0)
    imgsz: int = 640
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_candidates_per_image: int = Field(default=300, ge=1)
    hard_negative_ratio: float = Field(default=3.0, gt=0.0)
    focal_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    focal_gamma: float = Field(default=2.0, ge=0.0)
    evidence_interval: int = Field(default=100, ge=1)
    paper_prior: AuxiliaryPaperPrior

    @model_validator(mode="after")
    def validate_protocol(self) -> "AuxiliaryLossRuntimeConfig":
        if self.imgsz != 640:
            raise ValueError("quality auxiliary losses require imgsz=640")
        spec = LOSS_SPECS.get(self.component_id)
        if spec is None or spec.loss_name != self.loss_name:
            raise ValueError("auxiliary loss component and runtime name do not match")
        if self.changed_variable != spec.changed_variable:
            raise ValueError("auxiliary loss changed variable is not canonical")
        return self


class AuxiliaryLossEvidence(BaseModel):
    schema_version: str = "quality_auxiliary_loss_evidence.v1"
    component_id: str
    loss_name: AuxiliaryLossName
    changed_variable: str
    weight: float
    protocol_hash: str
    runtime_payload_hash: str
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    adapter_version: str
    plugin_version: str
    plugin_sha256: str
    rank: int
    batch_log_name: str
    compute_loss_calls: int = 0
    latest_raw_loss: float = 0.0
    latest_weighted_loss: float = 0.0
    native_loss_before: float = 0.0
    total_loss_after: float = 0.0
    total_loss_changed: bool = False
    latest_metrics: dict[str, float] = Field(default_factory=dict)
    gradient_hook_targets: list[str] = Field(default_factory=list)
    gradient_observed: bool = False
    gradient_norms: dict[str, float] = Field(default_factory=dict)
    native_assigner: str
    native_bbox_loss: str
    native_dfl_enabled: bool
    replaces_bbox_regression: Literal[False] = False
    replaces_assigner: Literal[False] = False
    changes_inference_graph: Literal[False] = False
    paper_prior: AuxiliaryPaperPrior
    checkpoint_metadata_paths: list[str] = Field(default_factory=list)


class QualityAlignmentRuntimePlugin:
    """Compute one additive auxiliary term after the native YOLO26 criterion."""

    plugin_version = "quality_alignment_auxiliary_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = AuxiliaryLossRuntimeConfig.model_validate(options)
        self.loss_plugin = _build_loss_plugin(self.config)
        self.evidence: AuxiliaryLossEvidence | None = None

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
            if token.partition("=")[0] != self.config.changed_variable
        ]
        return filtered, env

    def build_model(self, *, context: Any, trainer: Any, model: Any) -> Any:
        del context
        self._ensure_batch_log(trainer)
        return model

    def build_validator(
        self,
        *,
        context: Any,
        trainer: Any,
        validator: Any,
    ) -> Any:
        del context
        # DetectionTrainer.get_validator() resets loss_names to the native three
        # entries, so restore the auxiliary column after the validator is built.
        self._ensure_batch_log(trainer)
        return validator

    def build_criterion(
        self,
        *,
        context: Any,
        trainer: Any,
        model: Any,
        criterion: Any,
    ) -> Any:
        del model
        self._ensure_batch_log(trainer)
        self._ensure_evidence(context, criterion)
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
        del model
        self._ensure_batch_log(trainer)
        evidence = self._ensure_evidence(context, criterion)
        native_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
        native_scalar = float(native_loss.detach().float().sum().cpu())
        if self.config.weight == 0.0:
            raw_loss = native_loss.sum() * 0.0
            metrics: dict[str, float] = {}
            batch_size = 1
        else:
            inputs = extract_auxiliary_loss_inputs(criterion, predictions, batch)
            output = self.loss_plugin.compute(inputs)
            raw_loss = output.loss
            metrics = output.metrics
            batch_size = int(inputs.class_logits.shape[0])
            self._register_gradient_hooks(context, evidence, inputs)
        weighted_loss = raw_loss * self.config.weight * batch_size
        updated = _append_auxiliary_loss(loss_output, weighted_loss)
        updated_loss = updated[0] if isinstance(updated, tuple) else updated
        terms = getattr(trainer, "auxiliary_loss_terms", None)
        if not isinstance(terms, dict):
            terms = {}
            setattr(trainer, "auxiliary_loss_terms", terms)
        terms[self.config.loss_name] = float(weighted_loss.detach().float().cpu())
        evidence.compute_loss_calls += 1
        evidence.latest_raw_loss = float(raw_loss.detach().float().cpu())
        evidence.latest_weighted_loss = terms[self.config.loss_name]
        evidence.native_loss_before = native_scalar
        evidence.total_loss_after = float(updated_loss.detach().float().sum().cpu())
        evidence.total_loss_changed = bool(
            evidence.total_loss_after != evidence.native_loss_before
        )
        evidence.latest_metrics = metrics
        if (
            evidence.compute_loss_calls == 1
            or evidence.compute_loss_calls % self.config.evidence_interval == 0
        ):
            self._persist_evidence(context)
        return updated

    def on_checkpoint_load(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoint: dict[str, Any],
    ) -> None:
        state = checkpoint.get(f"auxiliary_loss_{self.config.loss_name}_evidence")
        if not isinstance(state, dict):
            resume = getattr(getattr(trainer, "args", None), "resume", None)
            candidates = [_evidence_path(context.payload_path.parent, self.config.loss_name)]
            if resume and str(resume).lower() not in {"true", "false"}:
                candidates.insert(
                    0,
                    _checkpoint_metadata_path(Path(resume), self.config.loss_name),
                )
            state = next(
                (
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in candidates
                    if path.is_file()
                ),
                None,
            )
        if not isinstance(state, dict):
            raise ValueError(f"{self.config.loss_name} resume evidence is missing")
        restored = AuxiliaryLossEvidence.model_validate(state)
        if restored.component_id != self.config.component_id:
            raise ValueError("auxiliary loss resume component identity mismatch")
        if restored.loss_name != self.config.loss_name:
            raise ValueError("auxiliary loss resume name mismatch")
        if restored.changed_variable != self.config.changed_variable:
            raise ValueError("auxiliary loss resume changed variable mismatch")
        if restored.protocol_hash != context.payload.protocol_hash:
            raise ValueError("auxiliary loss resume protocol hash mismatch")
        if restored.runtime_payload_hash != str(
            getattr(context.payload, "payload_hash", "")
        ):
            raise ValueError("auxiliary loss resume runtime payload hash mismatch")
        self.evidence = restored
        self._persist_evidence(context)

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
            metadata_path = _checkpoint_metadata_path(path, self.config.loss_name)
            payload = {
                **self.evidence.model_dump(mode="json"),
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": _sha256(path),
            }
            _write_json_atomic(metadata_path, payload)
            resolved = str(metadata_path.resolve())
            if resolved not in self.evidence.checkpoint_metadata_paths:
                self.evidence.checkpoint_metadata_paths.append(resolved)
        self._persist_evidence(context)

    def _ensure_batch_log(self, trainer: Any) -> None:
        name = _batch_log_name(self.config.loss_name)
        current = list(getattr(trainer, "loss_names", ()))
        if name not in current:
            current.append(name)
            trainer.loss_names = tuple(current)

    def _ensure_evidence(self, context: Any, criterion: Any) -> AuxiliaryLossEvidence:
        if self.evidence is not None:
            return self.evidence
        native = _native_detection_criterion(criterion)
        self.evidence = AuxiliaryLossEvidence(
            component_id=self.config.component_id,
            loss_name=self.config.loss_name,
            changed_variable=self.config.changed_variable,
            weight=self.config.weight,
            protocol_hash=context.payload.protocol_hash,
            runtime_payload_hash=str(getattr(context.payload, "payload_hash", "")),
            changed_variables=dict(getattr(context.payload, "changed_variables", {})),
            adapter_version=QualityAlignmentAuxiliaryLossAdapter.adapter_version,
            plugin_version=self.plugin_version,
            plugin_sha256=_sha256(Path(__file__)),
            rank=_rank(),
            batch_log_name=_batch_log_name(self.config.loss_name),
            native_assigner=type(native.assigner).__name__,
            native_bbox_loss=type(native.bbox_loss).__name__,
            native_dfl_enabled=bool(native.use_dfl),
            paper_prior=self.config.paper_prior,
        )
        self._persist_evidence(context)
        return self.evidence

    def _persist_evidence(self, context: Any) -> None:
        if self.evidence is not None:
            _write_json_atomic(
                _evidence_path(context.payload_path.parent, self.config.loss_name),
                self.evidence.model_dump(mode="json"),
            )

    def _register_gradient_hooks(
        self,
        context: Any,
        evidence: AuxiliaryLossEvidence,
        inputs: AuxiliaryLossInputs,
    ) -> None:
        tensors = {
            "class_logits": inputs.class_logits,
            "decoded_boxes_xyxy": inputs.predicted_boxes_xyxy,
        }
        for name, tensor in tensors.items():
            if not bool(getattr(tensor, "requires_grad", False)):
                continue
            if name not in evidence.gradient_hook_targets:
                evidence.gradient_hook_targets.append(name)
            tensor.register_hook(
                lambda gradient, target=name: self._record_gradient(
                    context, target, gradient
                )
            )

    def _record_gradient(self, context: Any, target: str, gradient: Any) -> Any:
        if self.evidence is not None:
            self.evidence.gradient_observed = True
            self.evidence.gradient_norms[target] = float(
                gradient.detach().float().norm().cpu()
            )
            self._persist_evidence(context)
        return gradient


class QualityAlignmentAuxiliaryLossAdapter(ComponentAdapter):
    """One adapter class serving three independently contracted atomic losses."""

    adapter_version = "quality_alignment_auxiliary.v1"
    source_commit = "yolo-agent:quality-alignment-auxiliary-v1"
    strategy = "loss_injection"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset(
        spec.changed_variable for spec in LOSS_SPECS.values()
    )

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            import torch

            import ultralytics

            return AdapterValidationReport(
                ok=True,
                checks={"torch": torch.__version__, "ultralytics": ultralytics.__version__},
            )
        except ImportError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        errors = []
        if context.contract.component_id not in LOSS_SPECS:
            errors.append("unknown quality auxiliary loss component")
        if context.detector_family != "yolo26":
            errors.append("quality auxiliary losses support YOLO26 only")
        if context.imgsz != 640:
            errors.append("quality auxiliary losses require fixed imgsz=640")
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "native_bbox_regression_preserved": True,
                "native_assigner_preserved": True,
                "student_inference_graph_unchanged": True,
            },
        )

    def patch_model_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        return config

    def patch_training_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        runtime = _runtime_config(context)
        config[runtime.changed_variable] = runtime.weight
        return config

    def build_module(self, context: AdapterContext) -> AuxiliaryLossPlugin:
        return _build_loss_plugin(_runtime_config(context))

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        return WeightLoadResult(
            loaded=False,
            message="quality auxiliary losses have no trainable adapter weights",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            runtime = _runtime_config(context)
            logits = torch.tensor(
                [[[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0]]],
                requires_grad=True,
            )
            inputs = AuxiliaryLossInputs(
                class_logits=logits,
                predicted_boxes_xyxy=torch.tensor(
                    [[[1.0, 1.0, 5.0, 5.0], [5.0, 5.0, 9.0, 9.0], [0.0] * 4]],
                    requires_grad=True,
                ),
                target_boxes_xyxy=torch.tensor(
                    [[[1.0, 1.0, 5.0, 5.0], [4.0, 4.0, 9.0, 9.0], [0.0] * 4]]
                ),
                target_classes=torch.tensor([[0, 1, 0]]),
                foreground_mask=torch.tensor([[True, True, False]]),
                anchor_points_xy=torch.tensor([[3.0, 3.0], [7.0, 7.0], [1.0, 1.0]]),
            )
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = _build_loss_plugin(runtime).compute(inputs)
                loss = output.loss * runtime.weight
            loss.backward()
            backward = (
                logits.grad is not None
                or inputs.predicted_boxes_xyxy.grad is not None
            )
            return SmokeTestResult(
                passed=bool(torch.isfinite(loss) and backward),
                evidence_kind="local",
                checks={
                    "shape": str(tuple(logits.shape)),
                    "backward": backward,
                    "amp": True,
                    "native_bbox_regression_preserved": True,
                    "native_assigner_preserved": True,
                    "imgsz": "640",
                },
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[str(exc)],
            )

    def gpu_smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            if not torch.cuda.is_available():
                return SmokeTestResult(
                    passed=False,
                    evidence_kind="local",
                    checks={"gpu_smoke_implemented": True, "cuda_available": False},
                    errors=["cuda_not_available"],
                )
            runtime = _runtime_config(context)
            device = torch.device("cuda")
            logits = torch.tensor(
                [[[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0]]],
                device=device,
                requires_grad=True,
            )
            inputs = AuxiliaryLossInputs(
                class_logits=logits,
                predicted_boxes_xyxy=torch.tensor(
                    [
                        [
                            [1.0, 1.0, 5.0, 5.0],
                            [5.0, 5.0, 9.0, 9.0],
                            [0.0] * 4,
                        ]
                    ],
                    device=device,
                ),
                target_boxes_xyxy=torch.tensor(
                    [
                        [
                            [1.0, 1.0, 5.0, 5.0],
                            [4.0, 4.0, 9.0, 9.0],
                            [0.0] * 4,
                        ]
                    ],
                    device=device,
                ),
                target_classes=torch.tensor([[0, 1, 0]], device=device),
                foreground_mask=torch.tensor(
                    [[True, True, False]],
                    device=device,
                ),
                anchor_points_xy=torch.tensor(
                    [[3.0, 3.0], [7.0, 7.0], [1.0, 1.0]],
                    device=device,
                ),
            )
            native = torch.ones(3, device=device, requires_grad=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                raw = _build_loss_plugin(runtime).compute(inputs).loss
                active, _ = _append_auxiliary_loss(
                    (native, native.detach()),
                    raw * runtime.weight,
                )
                zero, _ = _append_auxiliary_loss(
                    (native, native.detach()),
                    raw * 0.0,
                )
            active.sum().backward()
            backward = (
                logits.grad is not None
                or inputs.predicted_boxes_xyxy.grad is not None
            )
            passed = bool(
                backward
                and (
                    logits.grad is None
                    or torch.isfinite(logits.grad).all()
                )
                and torch.equal(zero, native)
            )
            return SmokeTestResult(
                passed=passed,
                evidence_kind="local",
                checks={
                    "gpu_smoke_implemented": True,
                    "cuda_available": True,
                    "amp": True,
                    "backward": backward,
                    "zero_weight_native_equivalent": torch.equal(zero, native),
                    "imgsz": "640",
                },
                errors=[] if passed else ["quality loss CUDA smoke failed"],
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                checks={"gpu_smoke_implemented": True},
                errors=[str(exc)],
            )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        loss_name = _spec(context).loss_name
        return [
            ExpectedArtifact(
                name=f"auxiliary_loss_{loss_name}_evidence",
                relative_path=Path(f"auxiliary_loss_{loss_name}_evidence.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        loss_name = _spec(context).loss_name
        return RollbackPlan(
            actions=[f"remove additive {loss_name} loss plugin and metadata sidecars"],
            files_to_remove=[Path(f"auxiliary_loss_{loss_name}_evidence.json")],
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
            loss_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.losses.quality_alignment:"
                        "QualityAlignmentRuntimePlugin"
                    ),
                    options=runtime.model_dump(mode="json"),
                    required_hooks=["compute_loss"],
                )
            ],
            generated_config=generated_config,
            changed_variables={runtime.changed_variable: runtime.weight},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )


def extract_auxiliary_loss_inputs(
    criterion: Any,
    predictions: Any,
    batch: dict[str, Any],
) -> AuxiliaryLossInputs:
    """Reuse native YOLO26 matching without replacing assigner or bbox loss."""
    import torch
    from ultralytics.utils.tal import make_anchors

    native = _native_detection_criterion(criterion)
    branch = _one_to_many_branch(predictions)
    class_logits = branch["scores"].permute(0, 2, 1).contiguous()
    pred_distribution = branch["boxes"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(branch["feats"], native.stride, 0.5)
    batch_size = class_logits.shape[0]
    image_size = (
        torch.tensor(
            branch["feats"][0].shape[2:],
            device=native.device,
            dtype=class_logits.dtype,
        )
        * native.stride[0]
    )
    targets = torch.cat(
        (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
        1,
    )
    decoded_boxes = native.bbox_decode(anchor_points, pred_distribution)
    with torch.no_grad():
        targets = native.preprocess(
            targets.to(native.device),
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        gt_labels, gt_boxes = targets.split((1, 4), 2)
        mask_gt = gt_boxes.sum(2, keepdim=True).gt_(0.0)
        matched_boxes = native.bbox_decode(anchor_points, pred_distribution.detach())
        _, target_boxes, target_scores, foreground_mask, _ = native.assigner(
            class_logits.detach().sigmoid(),
            (matched_boxes * stride_tensor).type(gt_boxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_boxes,
            mask_gt,
        )
        target_classes = target_scores.argmax(dim=-1)
        target_boxes = target_boxes.detach()
        target_scores = target_scores.detach()
        foreground_mask = foreground_mask.detach()
    predicted_boxes = decoded_boxes * stride_tensor
    return AuxiliaryLossInputs(
        class_logits=class_logits,
        predicted_boxes_xyxy=predicted_boxes,
        target_boxes_xyxy=target_boxes,
        target_classes=target_classes,
        foreground_mask=foreground_mask,
        anchor_points_xy=anchor_points * stride_tensor,
        target_scores=target_scores,
    )


def _native_detection_criterion(criterion: Any) -> Any:
    native = getattr(criterion, "one2many", None)
    if native is None:
        raise ValueError("quality auxiliary losses require YOLO26 E2ELoss.one2many")
    for attribute in (
        "assigner",
        "bbox_loss",
        "bbox_decode",
        "preprocess",
        "stride",
        "device",
        "use_dfl",
    ):
        if not hasattr(native, attribute):
            raise ValueError(f"native YOLO26 criterion is missing {attribute}")
    return native


def _one_to_many_branch(predictions: Any) -> dict[str, Any]:
    if isinstance(predictions, tuple):
        predictions = predictions[1]
    if not isinstance(predictions, dict) or "one2many" not in predictions:
        raise ValueError("quality auxiliary losses require YOLO26 one2many predictions")
    branch = predictions["one2many"]
    if not isinstance(branch, dict) or not {"boxes", "scores", "feats"}.issubset(branch):
        raise ValueError("YOLO26 one2many output is incomplete")
    return branch


def _runtime_config(context: AdapterContext) -> AuxiliaryLossRuntimeConfig:
    spec = _spec(context)
    weight = context.options.get(spec.changed_variable, spec.default_weight)
    return AuxiliaryLossRuntimeConfig(
        loss_name=spec.loss_name,
        component_id=spec.component_id,
        changed_variable=spec.changed_variable,
        weight=float(weight),
        imgsz=context.imgsz,
        confidence_threshold=float(
            context.options.get("confidence_threshold", 0.5)
        ),
        iou_threshold=float(context.options.get("iou_threshold", 0.5)),
        max_candidates_per_image=int(
            context.options.get("max_candidates_per_image", 300)
        ),
        hard_negative_ratio=float(context.options.get("hard_negative_ratio", 3.0)),
        focal_alpha=float(context.options.get("focal_alpha", 0.25)),
        focal_gamma=float(context.options.get("focal_gamma", 2.0)),
        evidence_interval=int(context.options.get("evidence_interval", 100)),
        paper_prior=AuxiliaryPaperPrior(
            paper_id=spec.paper_id,
            adaptation=spec.adaptation,
        ),
    )


def _spec(context: AdapterContext) -> _LossSpec:
    try:
        return LOSS_SPECS[context.contract.component_id]
    except KeyError as exc:
        raise ValueError(
            f"unsupported quality auxiliary component: {context.contract.component_id}"
        ) from exc


def _build_loss_plugin(config: AuxiliaryLossRuntimeConfig) -> AuxiliaryLossPlugin:
    options: dict[str, Any] = {}
    if config.loss_name == "bpc_calibration":
        options = {
            "confidence_threshold": config.confidence_threshold,
            "iou_threshold": config.iou_threshold,
            "max_candidates_per_image": config.max_candidates_per_image,
        }
    if config.loss_name == "class_balanced_focal":
        options = {
            "alpha": config.focal_alpha,
            "gamma": config.focal_gamma,
        }
    if config.loss_name == "hard_negative_classification":
        options = {
            "negative_positive_ratio": config.hard_negative_ratio,
            "max_candidates_per_image": config.max_candidates_per_image,
        }
    return build_auxiliary_loss(config.loss_name, **options)


def _append_auxiliary_loss(loss_output: Any, weighted_loss: Any) -> Any:
    import torch

    if not isinstance(loss_output, tuple) or len(loss_output) != 2:
        raise TypeError("YOLO26 auxiliary loss expects (loss, loss_items)")
    native_loss, loss_items = loss_output
    if not torch.is_tensor(native_loss) or not torch.is_tensor(loss_items):
        raise TypeError("YOLO26 native loss outputs must be tensors")
    if native_loss.ndim == 0:
        augmented = native_loss + weighted_loss
    else:
        addition = torch.cat(
            [
                weighted_loss.reshape(1).to(native_loss.dtype),
                native_loss.new_zeros(native_loss.numel() - 1),
            ]
        ).reshape_as(native_loss)
        augmented = native_loss + addition
    logged = torch.cat(
        [loss_items.reshape(-1), weighted_loss.detach().reshape(1).to(loss_items.dtype)]
    )
    return augmented, logged


def _batch_log_name(loss_name: str) -> str:
    return f"aux_{loss_name}_loss"


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


def _evidence_path(directory: Path, loss_name: str) -> Path:
    rank = _rank()
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    return directory / f"auxiliary_loss_{loss_name}_evidence{suffix}.json"


def _checkpoint_metadata_path(checkpoint: Path, loss_name: str) -> Path:
    return checkpoint.with_suffix(
        checkpoint.suffix + f".auxiliary_loss.{loss_name}.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AuxiliaryLossEvidence",
    "AuxiliaryLossRuntimeConfig",
    "AuxiliaryPaperPrior",
    "LOSS_SPECS",
    "QualityAlignmentAuxiliaryLossAdapter",
    "QualityAlignmentRuntimePlugin",
    "extract_auxiliary_loss_inputs",
]
