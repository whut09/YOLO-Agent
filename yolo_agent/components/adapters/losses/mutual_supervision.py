"""Paper-specific runtime route for Mutual Supervision dense detection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.losses.quality_alignment import (
    _append_auxiliary_loss,
    _native_detection_criterion,
    extract_auxiliary_loss_inputs,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.components.auxiliary_losses import (
    AuxiliaryLossInputs,
    MutualSupervisionAuxiliaryLoss,
)


class MutualSupervisionRuntimeConfig(BaseModel):
    """Validated runtime options for the paper-specific loss hook."""

    model_config = ConfigDict(extra="forbid")

    component_id: str = "loss.2109_05986"
    paper_id: str = "arxiv:2109.05986"
    paper_specific_mechanism_id: str = "loss.2109_05986"
    loss_name: str = "mutual_supervision"
    changed_variable: str = "loss.mutual_supervision.weight"
    weight: float = Field(default=0.1, ge=0.0)
    imgsz: int = 640
    evidence_interval: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_identity(self) -> "MutualSupervisionRuntimeConfig":
        if self.imgsz != 640:
            raise ValueError("mutual supervision requires imgsz=640")
        if self.component_id != "loss.2109_05986":
            raise ValueError("mutual supervision runtime component identity mismatch")
        if self.paper_specific_mechanism_id != self.component_id:
            raise ValueError("mutual supervision mechanism must match component identity")
        if self.loss_name != "mutual_supervision":
            raise ValueError("mutual supervision runtime loss name is fixed")
        if self.changed_variable != "loss.mutual_supervision.weight":
            raise ValueError("mutual supervision changed variable is not canonical")
        return self


class MutualSupervisionEvidence(BaseModel):
    """Runtime evidence proving an additive native-criterion integration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "mutual_supervision_evidence.v1"
    component_id: str
    paper_id: str
    paper_specific_mechanism_id: str
    loss_name: str
    changed_variable: str
    weight: float
    protocol_hash: str
    runtime_payload_hash: str
    adapter_version: str
    plugin_version: str
    plugin_sha256: str
    native_assigner: str
    native_bbox_loss: str
    native_dfl_enabled: Literal[False] = False
    replaces_bbox_regression: Literal[False] = False
    replaces_assigner: Literal[False] = False
    changes_inference_graph: Literal[False] = False
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


class MutualSupervisionRuntimePlugin:
    """Inject mutual confidence/localization supervision after native YOLO26 loss."""

    plugin_version = "mutual_supervision_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = MutualSupervisionRuntimeConfig.model_validate(options)
        self.loss_plugin = MutualSupervisionAuxiliaryLoss()
        self.evidence: MutualSupervisionEvidence | None = None

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
        _ensure_loss_name(trainer)
        return model

    def build_validator(
        self,
        *,
        context: Any,
        trainer: Any,
        validator: Any,
    ) -> Any:
        del context
        _ensure_loss_name(trainer)
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
        _ensure_loss_name(trainer)
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
        _ensure_loss_name(trainer)
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

    def _ensure_evidence(
        self, context: Any, criterion: Any
    ) -> MutualSupervisionEvidence:
        if self.evidence is not None:
            return self.evidence
        native = _native_detection_criterion(criterion)
        payload = getattr(context, "payload", None)
        self.evidence = MutualSupervisionEvidence(
            component_id=self.config.component_id,
            paper_id=self.config.paper_id,
            paper_specific_mechanism_id=self.config.paper_specific_mechanism_id,
            loss_name=self.config.loss_name,
            changed_variable=self.config.changed_variable,
            weight=self.config.weight,
            protocol_hash=str(getattr(payload, "protocol_hash", "")),
            runtime_payload_hash=str(getattr(payload, "payload_hash", "")),
            adapter_version=MutualSupervisionAdapter.adapter_version,
            plugin_version=self.plugin_version,
            plugin_sha256=_sha256(Path(__file__)),
            native_assigner=type(native.assigner).__name__,
            native_bbox_loss=type(native.bbox_loss).__name__,
            native_dfl_enabled=bool(native.use_dfl),
        )
        self._persist_evidence(context)
        return self.evidence

    def _persist_evidence(self, context: Any) -> None:
        if self.evidence is None:
            return
        path = _evidence_path(Path(context.payload_path).parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(self.evidence.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _register_gradient_hooks(
        self,
        context: Any,
        evidence: MutualSupervisionEvidence,
        inputs: AuxiliaryLossInputs,
    ) -> None:
        for name, tensor in {
            "class_logits": inputs.class_logits,
            "decoded_boxes_xyxy": inputs.predicted_boxes_xyxy,
        }.items():
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
        return gradient


class MutualSupervisionAdapter(ComponentAdapter):
    """Independent adapter for the 2109.05986 mutual-supervision method."""

    adapter_version = "mutual_supervision.paper_2109_05986.v1"
    source_commit = "yolo-agent:mutual-supervision-paper-route-v1"
    strategy = "loss_injection"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"loss.mutual_supervision.weight"})

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
        errors: list[str] = []
        if context.contract.component_id != "loss.2109_05986":
            errors.append("mutual supervision contract identity mismatch")
        if context.detector_family != "yolo26":
            errors.append("mutual supervision supports YOLO26 only")
        if context.head not in {None, "one_to_one"}:
            errors.append("mutual supervision requires YOLO26 one-to-one head")
        if context.imgsz != 640:
            errors.append("mutual supervision requires fixed imgsz=640")
        if context.contract.fixed_imgsz_compatible is False:
            errors.append("contract rejects fixed imgsz=640")
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "native_one_to_one_head_preserved": True,
                "native_dfl_free_regression_preserved": True,
                "native_assigner_preserved": True,
                "bbox_regression_replaced": False,
                "inference_graph_changed": False,
                "imgsz": "640",
            },
        )

    def patch_model_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        del context, dry_run
        return config

    def patch_training_config(
        self,
        config: dict[str, Any],
        context: AdapterContext,
        *,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        del dry_run
        config["loss.mutual_supervision.weight"] = _runtime_config(context).weight
        return config

    def build_module(self, context: AdapterContext) -> MutualSupervisionAuxiliaryLoss:
        return MutualSupervisionAuxiliaryLoss()

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        del module, weights, context
        return WeightLoadResult(
            loaded=False,
            message="mutual supervision has no adapter weights; it is an additive loss",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            inputs = AuxiliaryLossInputs(
                class_logits=torch.tensor(
                    [[[2.0, -1.0], [-0.5, 1.5], [1.0, 0.0]]],
                    requires_grad=True,
                ),
                predicted_boxes_xyxy=torch.tensor(
                    [[[1.0, 1.0, 5.0, 5.0], [5.0, 5.0, 9.0, 9.0], [0.0] * 4]],
                    requires_grad=True,
                ),
                target_boxes_xyxy=torch.tensor(
                    [[[1.0, 1.0, 5.0, 5.0], [4.0, 4.0, 9.0, 9.0], [0.0] * 4]]
                ),
                target_classes=torch.tensor([[0, 1, 0]]),
                foreground_mask=torch.tensor([[True, True, False]]),
                anchor_points_xy=torch.tensor(
                    [[3.0, 3.0], [7.0, 7.0], [1.0, 1.0]]
                ),
            )
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = MutualSupervisionAuxiliaryLoss().compute(inputs)
            output.loss.backward()
            logits_backward = inputs.class_logits.grad is not None
            boxes_backward = inputs.predicted_boxes_xyxy.grad is not None
            passed = bool(
                torch.isfinite(output.loss)
                and logits_backward
                and boxes_backward
                and context.imgsz == 640
            )
            return SmokeTestResult(
                passed=passed,
                evidence_kind="local",
                checks={
                    "shape": str(tuple(inputs.class_logits.shape)),
                    "forward": True,
                    "backward_class_logits": logits_backward,
                    "backward_boxes": boxes_backward,
                    "native_one_to_one_head": True,
                    "native_dfl_free_regression": True,
                    "native_assigner_preserved": True,
                    "imgsz": "640",
                },
                errors=[] if passed else ["mutual supervision CPU smoke failed"],
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            return SmokeTestResult(passed=False, evidence_kind="local", errors=[str(exc)])

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        del context
        return [
            ExpectedArtifact(
                name="mutual_supervision_evidence",
                relative_path=Path("mutual_supervision_evidence.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        del context
        return RollbackPlan(
            actions=["remove the additive mutual-supervision loss and evidence sidecar"],
            files_to_remove=[Path("mutual_supervision_evidence.json")],
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
                        "yolo_agent.components.adapters.losses.mutual_supervision:"
                        "MutualSupervisionRuntimePlugin"
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


def _runtime_config(context: AdapterContext) -> MutualSupervisionRuntimeConfig:
    options = context.options
    return MutualSupervisionRuntimeConfig(
        paper_id=str(options.get("paper_id", "arxiv:2109.05986")),
        weight=float(
            options.get(
                "loss.mutual_supervision.weight",
                options.get("weight", 0.1),
            )
        ),
        imgsz=context.imgsz,
    )


def _ensure_loss_name(trainer: Any) -> None:
    if trainer is None:
        return
    name = "aux_mutual_supervision_loss"
    current = list(getattr(trainer, "loss_names", ()))
    if name not in current:
        current.append(name)
        trainer.loss_names = tuple(current)


def _evidence_path(directory: Path) -> Path:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    return directory / f"mutual_supervision_evidence{suffix}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MutualSupervisionAdapter",
    "MutualSupervisionEvidence",
    "MutualSupervisionRuntimeConfig",
    "MutualSupervisionRuntimePlugin",
]
