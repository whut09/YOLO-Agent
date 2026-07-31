"""Component-specific assertions layered on the common real GPU contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.adapters.losses.quality_alignment import (
    AuxiliaryLossEvidence,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    DistillationEvidence,
)
from yolo_agent.components.adapters.head.p2_head import P2HeadManifest
from yolo_agent.components.adapters.neck.common import YOLO26NeckManifest
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingManifest,
)


GPUProfileValidator = Callable[
    [AdapterRuntimePayload, dict[str, Path]],
    dict[str, bool | str | int | float],
]


def validate_component_gpu_profile(
    component_id: str,
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    """Validate runtime facts that are unique to one adapter mechanism."""
    validator = _VALIDATORS.get(component_id)
    if validator is None:
        raise ValueError(f"GPU profile validator is not implemented: {component_id}")
    checks = validator(payload, artifacts)
    if not checks or not all(value is True for value in checks.values()):
        failed = sorted(name for name, value in checks.items() if value is not True)
        raise ValueError("component GPU profile failed: " + ", ".join(failed))
    return checks


def _validate_sampling(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    manifest = _json_model(
        SmallObjectSamplingManifest,
        artifacts,
        "adapter_sampler_manifest",
    )
    maximum = float(manifest.clipping_statistics.get("max_weight", 0.0))
    return {
        "sampling_payload_bound": manifest.runtime_payload_hash == payload.payload_hash,
        "sampling_protocol_bound": manifest.protocol_hash == payload.protocol_hash,
        "sampling_train_split_only": manifest.split == "train",
        "sampling_val_unchanged": manifest.val_unchanged,
        "sampling_weights_complete": bool(
            manifest.image_count > 0
            and manifest.sample_count > 0
            and len(manifest.raw_weights) == manifest.image_count
            and len(manifest.final_weights) == manifest.image_count
        ),
        "sampling_weights_bounded": bool(
            maximum >= 1.0
            and manifest.final_weights
            and max(manifest.final_weights) <= maximum
        ),
        "sampling_adapter_hash_recorded": len(manifest.adapter_hash) == 64,
    }


def _validate_quality_loss(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
    *,
    component_id: str,
    loss_name: str,
) -> dict[str, bool | str | int | float]:
    evidence = _json_model(
        AuxiliaryLossEvidence,
        artifacts,
        f"adapter_auxiliary_loss_{loss_name}_evidence",
    )
    metadata = [Path(item) for item in evidence.checkpoint_metadata_paths]
    return {
        "loss_component_bound": evidence.component_id == component_id,
        "loss_payload_bound": evidence.runtime_payload_hash == payload.payload_hash,
        "loss_protocol_bound": evidence.protocol_hash == payload.protocol_hash,
        "loss_compute_hook_observed": evidence.compute_loss_calls > 0,
        "loss_total_changed": bool(
            evidence.total_loss_changed and evidence.latest_weighted_loss != 0.0
        ),
        "loss_native_yolo26_preserved": bool(
            not evidence.replaces_bbox_regression
            and not evidence.replaces_assigner
            and not evidence.changes_inference_graph
            and not evidence.native_dfl_enabled
        ),
        "loss_checkpoint_metadata": bool(
            metadata and all(path.is_file() for path in metadata)
        ),
        "loss_paper_prior_not_exact": not evidence.paper_prior.exact_reproduction,
    }


def _validate_correlation_loss(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    return _validate_quality_loss(
        payload,
        artifacts,
        component_id="loss.quality.correlation",
        loss_name="correlation",
    )


def _validate_bpc_loss(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    return _validate_quality_loss(
        payload,
        artifacts,
        component_id="loss.calibration.bpc",
        loss_name="bpc_calibration",
    )


def _validate_pseudo_iou_loss(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    return _validate_quality_loss(
        payload,
        artifacts,
        component_id="loss.quality.pseudo_iou",
        loss_name="pseudo_iou",
    )


def _validate_distillation(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    evidence = _json_model(
        DistillationEvidence,
        artifacts,
        "adapter_distillation_evidence",
    )
    return {
        "distillation_payload_bound": evidence.runtime_payload_hash
        == payload.payload_hash,
        "distillation_protocol_bound": evidence.protocol_hash
        == payload.protocol_hash,
        "distillation_loss_observed": bool(
            evidence.compute_loss_calls > 0
            and evidence.total_loss_changed
            and evidence.latest_terms
        ),
        "distillation_teacher_safe": bool(
            evidence.teacher_eval
            and evidence.teacher_frozen
            and evidence.teacher_no_grad
        ),
        "distillation_shared_geometry": bool(
            evidence.shared_batch_tensor
            and evidence.geometry_policy == "shared_preprocessed_batch_tensor"
        ),
        "distillation_checkpoint_hashes": bool(
            len(evidence.teacher_checkpoint_sha256) == 64
            and len(evidence.student_checkpoint_sha256) == 64
        ),
        "distillation_resume_validated": evidence.resume_validated,
        "distillation_student_graph_unchanged": evidence.student_inference_graph_unchanged,
        "distillation_profiles_not_exact": all(
            not profile.exact_reproduction for profile in evidence.method_profiles
        ),
    }


def _validate_p2_head(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    manifest = _json_model(
        P2HeadManifest,
        artifacts,
        "adapter_p2_head_manifest",
    )
    return {
        "p2_payload_bound": manifest.runtime_payload_hash == payload.payload_hash,
        "p2_protocol_bound": manifest.protocol_hash == payload.protocol_hash,
        "p2_stride_four_observed": manifest.actual_tensor_strides == [4, 8, 16, 32],
        "p2_detection_path_integrated": bool(
            manifest.detect_input_count == 4
            and manifest.graph_integrated
            and manifest.detection_head_integrated
            and manifest.native_loss_integrated
        ),
        "p2_native_yolo26_preserved": bool(
            manifest.native_end2end
            and manifest.dfl_disabled
            and not manifest.external_nms_added
        ),
        "p2_partial_checkpoint_audited": bool(
            manifest.checkpoint_integrated
            and manifest.checkpoint.loaded
            and manifest.checkpoint.partial
            and manifest.checkpoint.matched_keys
            and manifest.checkpoint.newly_initialized_keys
            and len(manifest.checkpoint.checkpoint_sha256) == 64
        ),
        "p2_resource_guard_passed": manifest.resources.passed,
        "p2_generated_yaml_present": bool(
            artifacts.get("adapter_p2_model_yaml")
            and artifacts["adapter_p2_model_yaml"].is_file()
            and len(manifest.generated_yaml_sha256) == 64
        ),
    }


def _validate_neck(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
    *,
    component_id: str,
    neck_kind: str,
) -> dict[str, bool | str | int | float]:
    manifest = _json_model(
        YOLO26NeckManifest,
        artifacts,
        f"adapter_{component_id}_manifest",
    )
    plugin_adapter_hash = str(payload.model_graph_plugin[0].options["adapter_hash"])
    return {
        "neck_component_bound": manifest.component_id == component_id,
        "neck_kind_bound": manifest.neck_kind == neck_kind,
        "neck_protocol_bound": manifest.protocol_hash == payload.protocol_hash,
        "neck_adapter_hash_bound": manifest.adapter_hash == plugin_adapter_hash,
        "neck_stride_contract": bool(
            manifest.input_strides == [8, 16, 32]
            and manifest.output_strides == [8, 16, 32]
        ),
        "neck_native_yolo26_preserved": bool(
            manifest.native_end2end
            and manifest.dfl_disabled
            and not manifest.external_nms_added
        ),
        "neck_partial_checkpoint_audited": bool(
            manifest.checkpoint.loaded
            and manifest.checkpoint.partial
            and manifest.checkpoint.matched_keys
            and manifest.checkpoint.newly_initialized_keys
            and len(manifest.checkpoint.checkpoint_sha256) == 64
        ),
        "neck_resource_guard_passed": manifest.resources.passed,
        "neck_export_verified": manifest.export_dry_run,
        "neck_not_exact_reproduction": not manifest.exact_paper_reproduction,
    }


def _validate_multi_scale_neck(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    return _validate_neck(
        payload,
        artifacts,
        component_id="neck.multi_scale_fusion",
        neck_kind="multi_scale_fusion",
    )


def _validate_gold_neck(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    return _validate_neck(
        payload,
        artifacts,
        component_id="neck.gold_gather_distribute",
        neck_kind="gold_gather_distribute",
    )


def _validate_rtmdet_neck(
    payload: AdapterRuntimePayload,
    artifacts: dict[str, Path],
) -> dict[str, bool | str | int | float]:
    return _validate_neck(
        payload,
        artifacts,
        component_id="neck.rtmdet_large_kernel",
        neck_kind="rtmdet_large_kernel",
    )


def _json_model(
    model: type[Any],
    artifacts: dict[str, Path],
    key: str,
) -> Any:
    path = artifacts.get(key)
    if path is None or not path.is_file():
        raise ValueError(f"component GPU artifact missing: {key}")
    return model.model_validate_json(path.read_text(encoding="utf-8-sig"))


_VALIDATORS: dict[str, GPUProfileValidator] = {
    "sampling.small_object": _validate_sampling,
    "loss.quality.correlation": _validate_correlation_loss,
    "loss.calibration.bpc": _validate_bpc_loss,
    "loss.quality.pseudo_iou": _validate_pseudo_iou_loss,
    "distillation.yolo26_teacher_student": _validate_distillation,
    "head.p2_small_object": _validate_p2_head,
    "neck.multi_scale_fusion": _validate_multi_scale_neck,
    "neck.gold_gather_distribute": _validate_gold_neck,
    "neck.rtmdet_large_kernel": _validate_rtmdet_neck,
}


__all__ = ["GPUProfileValidator", "validate_component_gpu_profile"]
