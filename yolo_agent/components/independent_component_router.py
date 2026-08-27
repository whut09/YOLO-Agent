"""Independent YOLO26 paper-component routing.

Each listed component keeps its own contract identity, graph identity, and
queue eligibility. Assignment shadow cannot become an active train candidate
without shadow evidence. SAHI cannot become a training candidate.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import tempfile
from typing import Any, Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.components.adapters.audit_contract import (
    EXPECTED_RUNTIME_ADAPTERS,
    validate_audited_runtime_payload,
)
from yolo_agent.components.adapters import AdapterContext, ComponentAdapterRegistry
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.research.component_aliases import ComponentAliasResolver


IndependentComponentId = Literal[
    "assigner.optimal_transport",
    "assigner.task_aligned",
    "assigner.dynamic_smooth_label",
    "loss.quality.correlation",
    "loss.quality.pseudo_iou",
    "loss.calibration.bpc",
    "neck.gold_gather_distribute",
    "neck.multi_scale_fusion",
    "neck.rtmdet_large_kernel",
    "attention.spatial",
    "inference.sahi_slicing",
    "detection_head.task_aligned",
    "feature_pyramid.multi_scale",
]
QueueTrack = Literal["training", "inference", "blocked"]
ComponentDisposition = Literal[
    "queued",
    "evidence_recovery",
    "implementation_request",
    "blocked_runtime",
]


INDEPENDENT_COMPONENT_IDS: tuple[IndependentComponentId, ...] = (
    "assigner.optimal_transport",
    "assigner.task_aligned",
    "assigner.dynamic_smooth_label",
    "loss.quality.correlation",
    "loss.quality.pseudo_iou",
    "loss.calibration.bpc",
    "neck.gold_gather_distribute",
    "neck.multi_scale_fusion",
    "neck.rtmdet_large_kernel",
    "attention.spatial",
    "inference.sahi_slicing",
    "detection_head.task_aligned",
    "feature_pyramid.multi_scale",
)

GRAPH_IDENTITIES = {
    "assigner.optimal_transport": "assigner.optimal_transport",
    "assigner.task_aligned": "assigner.task_aligned",
    "assigner.dynamic_smooth_label": "assigner.dynamic_smooth_label",
    "neck.gold_gather_distribute": "neck.gold_gather_distribute",
    "neck.multi_scale_fusion": "neck.multi_scale_fusion",
    "neck.rtmdet_large_kernel": "neck.rtmdet_large_kernel",
    "attention.spatial": "attention.spatial",
    "detection_head.task_aligned": "detection_head.task_aligned",
    "feature_pyramid.multi_scale": "feature_pyramid.multi_scale",
}

ASSIGNMENT_SHADOW_COMPONENTS = {
    "assigner.optimal_transport",
    "assigner.task_aligned",
    "assigner.dynamic_smooth_label",
}

QUALITY_PAIR = ("loss.quality.correlation", "loss.quality.pseudo_iou")


class IndependentComponentRoute(BaseModel, YAMLModelMixin):
    model_config = ConfigDict(extra="forbid")

    component_id: IndependentComponentId
    recipe_id: str
    implementation_path: str
    adapter_class: str
    changed_variable: str
    runtime_hook: str
    runtime_payload_field: str
    evidence_artifact: str
    graph_identity: str
    inference_only: bool = False
    requires_shadow_evidence: bool = False
    paired_baseline_required: bool = True
    fixed_imgsz: int = 640
    yolo26_head_compatible: bool = True
    adapter_hash_required: bool = True
    asha_eligible: bool = False
    queue_track: QueueTrack
    disposition: ComponentDisposition
    reason_codes: list[str] = Field(default_factory=list)
    contract_maturity: str = "unknown"
    contract_can_execute: bool = False
    adapter_source_sha256: str | None = None
    runtime_payload_hash: str | None = None

    @model_validator(mode="after")
    def validate_route(self) -> "IndependentComponentRoute":
        if self.fixed_imgsz != 640:
            raise ValueError("independent paper components require imgsz=640")
        if not self.implementation_path or not self.adapter_class:
            raise ValueError("independent route requires an implementation path and adapter class")
        if not self.changed_variable or not self.runtime_hook:
            raise ValueError("independent route requires changed variable and runtime hook")
        if not self.runtime_payload_field or not self.evidence_artifact:
            raise ValueError("independent route requires runtime payload and evidence artifact")
        if self.inference_only and self.queue_track == "training":
            raise ValueError("inference-only components cannot be training candidates")
        if self.inference_only and self.asha_eligible:
            raise ValueError("inference-only components cannot enter training ASHA")
        if self.component_id in ASSIGNMENT_SHADOW_COMPONENTS and not self.requires_shadow_evidence:
            raise ValueError("assignment components require shadow evidence")
        return self


class IndependentComponentCoverage(BaseModel, YAMLModelMixin):
    schema_version: str = "independent_component_routing.v1"
    components_total: int
    routes: list[IndependentComponentRoute]
    swallowed_identities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_swallows(self) -> "IndependentComponentCoverage":
        if self.swallowed_identities:
            raise ValueError(f"independent identities were swallowed: {self.swallowed_identities}")
        if self.components_total != len(self.routes):
            raise ValueError("every independent component must have a route")
        ids = [item.component_id for item in self.routes]
        if len(set(ids)) != len(ids):
            raise ValueError("independent component routes must be unique")
        if "detection_head.task_aligned" in ids and "assigner.task_aligned" in ids:
            head = next(item for item in self.routes if item.component_id == "detection_head.task_aligned")
            assigner = next(item for item in self.routes if item.component_id == "assigner.task_aligned")
            if head.graph_identity == assigner.graph_identity:
                raise ValueError("task-aligned head cannot share assigner graph identity")
        graph_ids = [
            item.graph_identity
            for item in self.routes
            if item.component_id in {
                "neck.gold_gather_distribute",
                "neck.multi_scale_fusion",
                "neck.rtmdet_large_kernel",
                "feature_pyramid.multi_scale",
            }
        ]
        if len(set(graph_ids)) != 4:
            raise ValueError("neck and pyramid graph identities must stay distinct")
        return self


class IndependentComponentRouter:
    """Route one independent paper component without collapsing identities."""

    def route(
        self,
        component_id: IndependentComponentId,
        *,
        has_payload: bool = False,
        has_changed_variable: bool = False,
        has_evidence: bool = False,
        has_shadow_evidence: bool = False,
        has_adapter_hash: bool = False,
        imgsz: int = 640,
        yolo26_head_compatible: bool = True,
        paired_baseline: bool = False,
        contract: ComponentContract | None = None,
        contract_can_execute: bool | None = None,
        adapter_source_sha256: str | None = None,
        runtime_payload_hash: str | None = None,
    ) -> IndependentComponentRoute:
        catalog = COMPONENT_CATALOG[component_id]
        reasons: list[str] = []
        if imgsz != 640:
            reasons.append("fixed_imgsz_640_required")
        if not yolo26_head_compatible:
            reasons.append("yolo26_head_incompatible")
        if not paired_baseline:
            reasons.append("paired_baseline_required")
        executable = contract_can_execute if contract_can_execute is not None else contract is None
        if not executable:
            reasons.append("contract_execution_gate_not_satisfied")
        if not has_payload:
            reasons.append("runtime_payload_missing")
        if not has_changed_variable:
            reasons.append("changed_variable_missing")
        if not has_evidence:
            reasons.append("evidence_artifact_missing")
        if not has_adapter_hash:
            reasons.append("adapter_hash_missing")
        if catalog["requires_shadow_evidence"] and not has_shadow_evidence:
            reasons.append("assignment_shadow_evidence_required")
        if catalog["implementation_request"]:
            reasons.append(f"adapter_required:{component_id}")

        inference_only = component_id == "inference.sahi_slicing"
        if inference_only:
            track: QueueTrack = "inference"
            disposition: ComponentDisposition = "blocked_runtime" if reasons else "queued"
            reasons.append("inference_only_not_training_candidate")
            asha = False
        elif reasons:
            track = "blocked"
            disposition = (
                "implementation_request"
                if any(item.startswith("adapter_required") for item in reasons)
                else "evidence_recovery"
                if "assignment_shadow_evidence_required" in reasons
                or "runtime_payload_missing" in reasons
                or "changed_variable_missing" in reasons
                or "evidence_artifact_missing" in reasons
                or "contract_execution_gate_not_satisfied" in reasons
                else "blocked_runtime"
            )
            asha = False
        else:
            track = "training"
            disposition = "queued"
            asha = True

        return IndependentComponentRoute(
            component_id=component_id,
            recipe_id=catalog["recipe_id"],
            implementation_path=catalog["implementation_path"],
            adapter_class=catalog["adapter_class"],
            changed_variable=catalog["changed_variable"],
            runtime_hook=catalog["runtime_hook"],
            runtime_payload_field=catalog["runtime_payload_field"],
            evidence_artifact=catalog["evidence_artifact"],
            graph_identity=catalog["graph_identity"],
            inference_only=inference_only,
            requires_shadow_evidence=catalog["requires_shadow_evidence"],
            asha_eligible=asha,
            queue_track=track,
            disposition=disposition,
            reason_codes=list(dict.fromkeys(reasons)),
            contract_maturity=contract.maturity if contract else "unknown",
            contract_can_execute=executable,
            adapter_source_sha256=adapter_source_sha256,
            runtime_payload_hash=runtime_payload_hash,
        )

    def coverage(self, **kwargs: Any) -> IndependentComponentCoverage:
        return IndependentComponentCoverage(
            components_total=len(INDEPENDENT_COMPONENT_IDS),
            routes=[self.route(component_id, **kwargs) for component_id in INDEPENDENT_COMPONENT_IDS],
        )

    def audit_coverage(
        self,
        *,
        workspace: Path | str | None = None,
        protocol_hash: str = "independent-component-cpu-audit",
        has_evidence: bool = False,
        has_shadow_evidence: bool = False,
        paired_baseline: bool = False,
    ) -> IndependentComponentCoverage:
        """Audit every declared identity without granting training eligibility."""
        return IndependentComponentCoverage(
            components_total=len(INDEPENDENT_COMPONENT_IDS),
            routes=[
                self.audit(
                    component_id,
                    workspace=workspace,
                    protocol_hash=protocol_hash,
                    has_evidence=has_evidence,
                    has_shadow_evidence=has_shadow_evidence,
                    paired_baseline=paired_baseline,
                )
                for component_id in INDEPENDENT_COMPONENT_IDS
            ],
        )

    @staticmethod
    def recipe_binding_reasons(recipe_id: str, component_ids: list[str]) -> list[str]:
        """Return identity errors without treating a generic recipe as paper-specific."""
        reasons: list[str] = []
        if len(component_ids) != 1:
            return reasons
        component_id = component_ids[0]
        catalog = COMPONENT_CATALOG.get(component_id)  # type: ignore[arg-type]
        if catalog is None:
            return reasons
        expected = str(catalog["recipe_id"])
        if recipe_id != expected:
            reasons.append(
                f"independent_recipe_binding_mismatch:{component_id}:{expected}"
            )
        if component_id == "inference.sahi_slicing" and recipe_id != "sahi_slicing_inference":
            reasons.append("sahi_inference_recipe_identity_required")
        return reasons

    def audit(
        self,
        component_id: IndependentComponentId,
        *,
        workspace: Path | str | None = None,
        protocol_hash: str = "independent-component-cpu-audit",
        has_evidence: bool = False,
        has_shadow_evidence: bool = False,
        paired_baseline: bool = False,
    ) -> IndependentComponentRoute:
        """Run the non-GPU contract/payload smoke before queue classification.

        This deliberately does not certify a component for ASHA. Contract
        maturity and paired evidence remain separate promotion gates.
        """
        resolver = ComponentAliasResolver.from_yaml()
        contract = resolver.contracts.get(component_id)
        if contract is None:
            return self.route(
                component_id,
                has_payload=False,
                has_changed_variable=False,
                has_evidence=has_evidence,
                has_shadow_evidence=has_shadow_evidence,
                paired_baseline=paired_baseline,
            )
        catalog = COMPONENT_CATALOG[component_id]
        reasons: list[str] = []
        source_hash: str | None = None
        payload_hash: str | None = None
        payload_ok = False
        changed_ok = False
        try:
            module = importlib.import_module(str(contract.implementation_path))
            adapter_type = getattr(module, str(contract.adapter_class))
            if not inspect.isclass(adapter_type):
                raise TypeError("adapter_class is not a class")
            if str(contract.implementation_path) != catalog["implementation_path"]:
                reasons.append("implementation_path_mismatch")
            if str(contract.adapter_class) != catalog["adapter_class"]:
                reasons.append("adapter_class_mismatch")
            if contract.changed_variable != catalog["changed_variable"]:
                reasons.append("changed_variable_contract_mismatch")
            source_path = Path(inspect.getfile(adapter_type)).resolve()
            source_hash = _sha256_file(source_path)
            with tempfile.TemporaryDirectory(
                prefix="yolo26-independent-audit-",
                dir=workspace,
            ) as temp_dir:
                context = AdapterContext(
                    contract=contract,
                    detector_family="yolo26",
                    head="one_to_one",
                    imgsz=640,
                    workspace=Path(temp_dir),
                )
                adapter = ComponentAdapterRegistry().create_for_contract(contract)
                smoke = adapter.smoke_test(context)
                if not smoke.passed:
                    reasons.append("cpu_smoke_failed")
                payload = adapter.build_runtime_payload(
                    context,
                    protocol_hash=protocol_hash,
                    base_command=["python", "-m", "yolo_agent.adapters.ultralytics.runtime_entrypoint"],
                    generated_config={"imgsz": 640},
                )
                if payload is None:
                    reasons.append("runtime_payload_missing")
                else:
                    payload.verify_imports()
                    validate_audited_runtime_payload(payload, component_id)
                    payload_hash = payload.payload_hash
                    payload_ok = component_id in payload.component_ids
                    changed_ok = catalog["changed_variable"] in payload.changed_variables
                    if not payload_ok:
                        reasons.append("runtime_payload_component_missing")
                    if not changed_ok:
                        reasons.append("changed_variable_missing")
        except (AttributeError, ImportError, ModuleNotFoundError, OSError, TypeError, ValueError) as exc:
            reasons.append(f"runtime_probe_failed:{type(exc).__name__}")
        route = self.route(
            component_id,
            has_payload=payload_ok,
            has_changed_variable=changed_ok,
            has_evidence=has_evidence,
            has_shadow_evidence=has_shadow_evidence,
            has_adapter_hash=source_hash is not None,
            paired_baseline=paired_baseline,
            contract=contract,
            contract_can_execute=contract.can_execute,
            adapter_source_sha256=source_hash,
            runtime_payload_hash=payload_hash,
        )
        return route.model_copy(update={
            "reason_codes": list(dict.fromkeys([*route.reason_codes, *reasons])),
            "disposition": (
                "implementation_request"
                if any(item.startswith("runtime_probe_failed") for item in reasons)
                else route.disposition
            ),
            "queue_track": (
                "blocked"
                if any(item.startswith("runtime_probe_failed") for item in reasons)
                else route.queue_track
            ),
            "asha_eligible": route.asha_eligible and not reasons,
        })


COMPONENT_CATALOG: dict[IndependentComponentId, dict[str, Any]] = {
    "assigner.optimal_transport": {
        "recipe_id": "yolo26_ota_assignment_shadow",
        "implementation_path": "yolo_agent.components.adapters.assigners.yolo26_assignment",
        "adapter_class": "YOLO26AssignmentAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["assigner.optimal_transport"].changed_variable,
        "runtime_hook": "compute_loss",
        "runtime_payload_field": "assigner_plugin",
        "evidence_artifact": "assignment_ota_shadow_evidence.json",
        "graph_identity": "assigner.optimal_transport",
        "requires_shadow_evidence": True,
        "implementation_request": False,
    },
    "assigner.task_aligned": {
        "recipe_id": "yolo26_tood_tal_assignment_shadow",
        "implementation_path": "yolo_agent.components.adapters.assigners.yolo26_assignment",
        "adapter_class": "YOLO26AssignmentAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["assigner.task_aligned"].changed_variable,
        "runtime_hook": "compute_loss",
        "runtime_payload_field": "assigner_plugin",
        "evidence_artifact": "assignment_tood_tal_shadow_evidence.json",
        "graph_identity": "assigner.task_aligned",
        "requires_shadow_evidence": True,
        "implementation_request": False,
    },
    "assigner.dynamic_smooth_label": {
        "recipe_id": "yolo26_dsla_assignment_shadow",
        "implementation_path": "yolo_agent.components.adapters.assigners.yolo26_assignment",
        "adapter_class": "YOLO26AssignmentAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["assigner.dynamic_smooth_label"].changed_variable,
        "runtime_hook": "compute_loss",
        "runtime_payload_field": "assigner_plugin",
        "evidence_artifact": "assignment_dsla_shadow_evidence.json",
        "graph_identity": "assigner.dynamic_smooth_label",
        "requires_shadow_evidence": True,
        "implementation_request": False,
    },
    "loss.quality.correlation": {
        "recipe_id": "yolo26_correlation_auxiliary_loss",
        "implementation_path": "yolo_agent.components.adapters.losses.quality_alignment",
        "adapter_class": "QualityAlignmentAuxiliaryLossAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["loss.quality.correlation"].changed_variable,
        "runtime_hook": "compute_loss",
        "runtime_payload_field": "loss_plugin",
        "evidence_artifact": "auxiliary_loss_correlation_evidence.json",
        "graph_identity": "loss.quality.correlation",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "loss.quality.pseudo_iou": {
        "recipe_id": "yolo26_pseudo_iou_quality_auxiliary_loss",
        "implementation_path": "yolo_agent.components.adapters.losses.quality_alignment",
        "adapter_class": "QualityAlignmentAuxiliaryLossAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["loss.quality.pseudo_iou"].changed_variable,
        "runtime_hook": "compute_loss",
        "runtime_payload_field": "loss_plugin",
        "evidence_artifact": "auxiliary_loss_pseudo_iou_evidence.json",
        "graph_identity": "loss.quality.pseudo_iou",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "loss.calibration.bpc": {
        "recipe_id": "yolo26_bpc_calibration_auxiliary_loss",
        "implementation_path": "yolo_agent.components.adapters.losses.quality_alignment",
        "adapter_class": "QualityAlignmentAuxiliaryLossAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["loss.calibration.bpc"].changed_variable,
        "runtime_hook": "compute_loss",
        "runtime_payload_field": "loss_plugin",
        "evidence_artifact": "auxiliary_loss_bpc_calibration_evidence.json",
        "graph_identity": "loss.calibration.bpc",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "neck.gold_gather_distribute": {
        "recipe_id": "yolo26_gold_gather_distribute_neck",
        "implementation_path": "yolo_agent.components.adapters.neck.gold_gd_adapter",
        "adapter_class": "GoldGatherDistributeAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["neck.gold_gather_distribute"].changed_variable,
        "runtime_hook": "build_model",
        "runtime_payload_field": "model_graph_plugin",
        "evidence_artifact": "neck_gold_gather_distribute_manifest.json",
        "graph_identity": "neck.gold_gather_distribute",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "neck.multi_scale_fusion": {
        "recipe_id": "yolo26_generic_multi_scale_fusion",
        "implementation_path": "yolo_agent.components.adapters.neck.multi_scale_adapter",
        "adapter_class": "MultiScaleFusionAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["neck.multi_scale_fusion"].changed_variable,
        "runtime_hook": "build_model",
        "runtime_payload_field": "model_graph_plugin",
        "evidence_artifact": "neck_multi_scale_fusion_manifest.json",
        "graph_identity": "neck.multi_scale_fusion",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "neck.rtmdet_large_kernel": {
        "recipe_id": "yolo26_rtmdet_large_kernel_neck",
        "implementation_path": "yolo_agent.components.adapters.neck.rtmdet_adapter",
        "adapter_class": "RTMDetLargeKernelNeckAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["neck.rtmdet_large_kernel"].changed_variable,
        "runtime_hook": "build_model",
        "runtime_payload_field": "model_graph_plugin",
        "evidence_artifact": "neck_rtmdet_large_kernel_manifest.json",
        "graph_identity": "neck.rtmdet_large_kernel",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "attention.spatial": {
        "recipe_id": "yolo26_spatial_attention",
        "implementation_path": "yolo_agent.components.adapters.neck.component_adapters",
        "adapter_class": "SpatialAttentionAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["attention.spatial"].changed_variable,
        "runtime_hook": "build_model",
        "runtime_payload_field": "model_graph_plugin",
        "evidence_artifact": "neck_spatial_attention_manifest.json",
        "graph_identity": "attention.spatial",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "inference.sahi_slicing": {
        "recipe_id": "sahi_slicing_inference",
        "implementation_path": "yolo_agent.components.adapters.inference.slicing",
        "adapter_class": "SlicingInferenceAdapter",
        "changed_variable": EXPECTED_RUNTIME_ADAPTERS["inference.sahi_slicing"].changed_variable,
        "runtime_hook": "prepare_command",
        "runtime_payload_field": "inference_plugin",
        "evidence_artifact": "slicing_inference_protocol.json",
        "graph_identity": "inference.sahi_slicing",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "detection_head.task_aligned": {
        "recipe_id": "yolo26_task_aligned_head",
        "implementation_path": "yolo_agent.components.adapters.heads.task_aligned",
        "adapter_class": "TaskAlignedHeadAdapter",
        "changed_variable": "model.task_aligned_head",
        "runtime_hook": "build_model",
        "runtime_payload_field": "model_graph_plugin",
        "evidence_artifact": "task_aligned_head_manifest.json",
        "graph_identity": "detection_head.task_aligned",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
    "feature_pyramid.multi_scale": {
        "recipe_id": "yolo26_feature_pyramid_multi_scale",
        "implementation_path": "yolo_agent.components.adapters.neck.feature_pyramid_adapter",
        "adapter_class": "FeaturePyramidMultiScaleAdapter",
        "changed_variable": "model.feature_pyramid",
        "runtime_hook": "build_model",
        "runtime_payload_field": "model_graph_plugin",
        "evidence_artifact": "feature_pyramid_multi_scale_manifest.json",
        "graph_identity": "feature_pyramid.multi_scale",
        "requires_shadow_evidence": False,
        "implementation_request": False,
    },
}


def default_independent_component_router() -> IndependentComponentRouter:
    return IndependentComponentRouter()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
