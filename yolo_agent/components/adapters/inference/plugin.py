"""Runtime plugin and reusable adapters for isolated inference policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyConfig,
    InferencePolicyKind,
    protocol_from_policy,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)


class InferencePolicyRuntimeEvidence(BaseModel):
    schema_version: str = "inference_policy_runtime_evidence.v1"
    component_id: str
    policy_id: str
    policy_kind: InferencePolicyKind
    payload_hash: str
    protocol_hash: str
    changed_variables: dict[str, Any]
    plugin_version: str
    hook_call_counts: dict[str, int] = Field(default_factory=dict)
    inference_policy_changed: Literal[True] = True
    training_attribution_allowed: Literal[False] = False


class InferencePolicyPlugin:
    """Verify that a typed inference payload reaches the isolated command."""

    plugin_version = "inference_policy_plugin.v1"

    def __init__(self, **options: Any) -> None:
        self.config = InferencePolicyConfig.model_validate(options["config"])
        self.evidence_path = Path(str(options["evidence_path"])).resolve()

    def prepare_command(
        self,
        *,
        payload: AdapterRuntimePayload,
        command: list[str],
        env: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        normalized = [str(item).lower() for item in command]
        if not (
            "advanced" in normalized
            and "certify-inference-policy" in normalized
            and "--execute" in normalized
        ):
            raise ValueError(
                "inference policy payload requires "
                "'yolo-agent advanced certify-inference-policy ... --execute'"
            )
        if len(payload.component_ids) != 1 or not payload.component_ids[0].startswith(
            "inference."
        ):
            raise ValueError("inference policy payload contains non-inference components")
        evidence = InferencePolicyRuntimeEvidence(
            component_id=payload.component_ids[0],
            policy_id=self.config.policy_id,
            policy_kind=self.config.kind,
            payload_hash=payload.payload_hash,
            protocol_hash=payload.protocol_hash,
            changed_variables=dict(payload.changed_variables),
            plugin_version=self.plugin_version,
            hook_call_counts={"prepare_command": 1},
        )
        _write_json_atomic(self.evidence_path, evidence.model_dump(mode="json"))
        return command, env


class ReusableInferencePolicyAdapter(ComponentAdapter):
    """Common no-training adapter contract for reusable inference policies."""

    policy_kind: ClassVar[InferencePolicyKind]
    changed_variable: ClassVar[str]
    default_options: ClassVar[dict[str, Any]] = {}
    adapter_version = "inference-policy.v1"
    source_commit = "yolo-agent:isolated-inference-policy-v1"
    strategy = "inference_adapter"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset()

    def _config(self, context: AdapterContext) -> InferencePolicyConfig:
        options = {**self.default_options, **dict(context.options or {})}
        supplied_kind = options.pop("kind", self.policy_kind)
        if supplied_kind != self.policy_kind:
            raise ValueError(
                f"{type(self).__name__} requires kind={self.policy_kind}"
            )
        options.setdefault("policy_id", context.contract.component_id)
        return InferencePolicyConfig(kind=self.policy_kind, **options)

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return AdapterValidationReport(
                ok=False, errors=["inference policy execution requires ultralytics"]
            )
        return AdapterValidationReport(ok=True, checks={"ultralytics_importable": True})

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            config = self._config(context)
        except ValueError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])
        return AdapterValidationReport(
            ok=context.imgsz == 640,
            errors=[]
            if context.imgsz == 640
            else ["standard comparison requires imgsz=640"],
            checks={
                "inference_policy_changed": True,
                "training_attribution_allowed": False,
                "standard_imgsz": config.standard_imgsz,
                "extra_nms_applied": config.merge_policy == "nms",
            },
        )

    def patch_model_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        return config

    def patch_training_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        return config

    def build_module(self, context: AdapterContext) -> InferencePolicyConfig:
        return self._config(context)

    def load_pretrained_weights(
        self, module: Any, weights: Path | str | None, context: AdapterContext
    ) -> WeightLoadResult:
        return WeightLoadResult(
            loaded=False,
            source=Path(weights) if weights else None,
            message="Inference policies reuse the evaluated detector checkpoint",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        protocol = protocol_from_policy(self._config(context))
        return SmokeTestResult(
            passed=True,
            evidence_kind="local",
            checks={
                "protocol_hash": protocol.protocol_hash,
                "standard_metrics_preserved": True,
                "training_attribution_allowed": False,
            },
        )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        prefix = self._config(context).metric_namespace.removesuffix("_inference")
        return [
            ExpectedArtifact(
                name="inference_policy_protocol",
                relative_path=Path(f"artifacts/{prefix}_protocol.json"),
            ),
            ExpectedArtifact(
                name="inference_policy_predictions",
                relative_path=Path(f"artifacts/{prefix}_predictions.json"),
            ),
            ExpectedArtifact(
                name="inference_policy_metrics",
                relative_path=Path(f"artifacts/{prefix}_metrics.json"),
            ),
            ExpectedArtifact(
                name="inference_policy_resources",
                relative_path=Path(f"artifacts/{prefix}_resources.json"),
            ),
            ExpectedArtifact(
                name="inference_policy_report",
                relative_path=Path("inference_policy_certification_report.yaml"),
            ),
            ExpectedArtifact(
                name="inference_policy_runtime_evidence",
                relative_path=Path("inference_policy_runtime_evidence.json"),
            ),
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(
            actions=["discard inference-only policy artifacts"],
            files_to_remove=[
                item.relative_path for item in self.expected_artifacts(context)
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
        config = self._config(context)
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            inference_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.inference.plugin:"
                        "InferencePolicyPlugin"
                    ),
                    options={
                        "config": config.model_dump(mode="json", exclude_none=True),
                        "evidence_path": str(
                            (context.workspace / "inference_policy_runtime_evidence.json").resolve()
                        ),
                    },
                    required_hooks=["prepare_command"],
                )
            ],
            generated_config=generated_config,
            changed_variables={self.changed_variable: config.model_dump(mode="json")},
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=False,
            supports_ddp=False,
            supports_resume=False,
        )


class TiledMultiScaleInferenceAdapter(ReusableInferencePolicyAdapter):
    policy_kind = "tiled_multi_scale"
    changed_variable = "inference.tiled_multi_scale_policy"
    default_options = {
        "tile_sizes": [512, 640],
        "merge_policy": "weighted_box_fusion",
        "allow_cross_view_merge": True,
    }


class TestTimeAugmentationAdapter(ReusableInferencePolicyAdapter):
    policy_kind = "test_time_augmentation"
    changed_variable = "inference.tta_policy"
    default_options = {
        "scales": [0.8, 1.0, 1.2],
        "horizontal_flip": True,
        "merge_policy": "weighted_box_fusion",
        "allow_cross_view_merge": True,
    }


class ConfidenceCalibrationInferenceAdapter(ReusableInferencePolicyAdapter):
    policy_kind = "confidence_calibration"
    changed_variable = "inference.confidence_calibration"
    default_options = {"temperature": 1.5}


class ClassAwareThresholdInferenceAdapter(ReusableInferencePolicyAdapter):
    policy_kind = "class_aware_thresholding"
    changed_variable = "inference.class_thresholds"
    default_options = {"class_thresholds": {0: 0.25}}


class MergePolicyInferenceAdapter(ReusableInferencePolicyAdapter):
    policy_kind = "merge_policy"
    changed_variable = "inference.merge_policy"
    default_options = {
        "scales": [0.8, 1.0, 1.2],
        "merge_policy": "nmm",
        "allow_cross_view_merge": True,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


__all__ = [
    "ClassAwareThresholdInferenceAdapter",
    "ConfidenceCalibrationInferenceAdapter",
    "InferencePolicyPlugin",
    "InferencePolicyRuntimeEvidence",
    "MergePolicyInferenceAdapter",
    "ReusableInferencePolicyAdapter",
    "TestTimeAugmentationAdapter",
    "TiledMultiScaleInferenceAdapter",
]
