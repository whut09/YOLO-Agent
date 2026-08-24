"""Generate the per-paper execution requirements matrix from the inventory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import yaml

from yolo_agent.components.adapters.distillation.method_registry import (
    DistillationMethodRegistry,
)
from yolo_agent.research.paper_execution_requirement_schemas import (
    PaperExecutionRequirement,
    PaperExecutionRequirementsMatrix,
)
from yolo_agent.research.paper_execution_schemas import (
    PaperExecutionInventory,
    PaperExecutionSpec,
)
from yolo_agent.research.paper_protocol_catalog import build_paper_protocol_contract
from yolo_agent.research.paper_protocol_contract import PaperProtocolContract


_GENERIC = {
    "distillation.yolo26_teacher_student",
    "domain_adaptation.general",
    "quality_alignment.general",
}

_STANDARD_ROUTES: dict[str, dict[str, Any]] = {
    "assigner.optimal_transport": {
        "adapter": "assigner.optimal_transport",
        "changed": ["matching"],
        "payload": {"assignment_path": "one_to_many", "mode": "shadow", "native_output": "five_tensor"},
        "recipes": ["yolo26_ota_assignment_shadow"],
    },
    "assigner.task_aligned": {
        "adapter": "assigner.task_aligned",
        "changed": ["matching", "task_aligned_score"],
        "payload": {"assignment_path": "one_to_many", "mode": "shadow", "native_output": "five_tensor"},
        "recipes": ["yolo26_tood_tal_assignment_shadow"],
    },
    "assigner.dynamic_smooth_label": {
        "adapter": "assigner.dynamic_smooth_label",
        "changed": ["label_smoothing"],
        "payload": {"mode": "shadow", "native_output": "five_tensor"},
        "recipes": ["yolo26_dsla_assignment_shadow"],
    },
    "detection_head.task_aligned": {
        "adapter": "detection_head.task_aligned",
        "changed": ["head.task_aligned_weighting"],
        "payload": {"head_mode": "native_yolo26_one_to_one"},
        "recipes": ["yolo26_task_aligned_head"],
    },
    "loss.quality.correlation": {
        "adapter": "loss.quality.correlation",
        "changed": ["loss.correlation.weight"],
        "payload": {"loss_name": "correlation", "hook": "compute_loss"},
        "recipes": ["yolo26_correlation_auxiliary_loss"],
    },
    "loss.quality.pseudo_iou": {
        "adapter": "loss.quality.pseudo_iou",
        "changed": ["loss.pseudo_iou.weight"],
        "payload": {"loss_name": "pseudo_iou", "hook": "compute_loss"},
        "recipes": ["yolo26_pseudo_iou_quality_auxiliary_loss"],
    },
    "loss.calibration.bpc": {
        "adapter": "loss.calibration.bpc",
        "changed": ["loss.bpc.weight"],
        "payload": {"loss_name": "bpc", "hook": "compute_loss"},
        "recipes": ["yolo26_bpc_calibration_auxiliary_loss"],
    },
    "neck.rtmdet_large_kernel": {
        "adapter": "neck.rtmdet_large_kernel",
        "changed": ["neck.rtmdet_large_kernel.enabled"],
        "payload": {"graph_identity": "rtmdet_large_kernel_neck", "preserves_one_to_one_head": True},
        "recipes": ["yolo26_rtmdet_large_kernel_neck"],
    },
    "neck.gold_gather_distribute": {
        "adapter": "neck.gold_gather_distribute",
        "changed": ["neck.gold_gather_distribute.enabled"],
        "payload": {"graph_identity": "gold_gather_distribute"},
        "recipes": ["yolo26_gold_gather_distribute_neck"],
    },
    "neck.multi_scale_fusion": {
        "adapter": "neck.multi_scale_fusion",
        "changed": ["neck.multi_scale_fusion.enabled"],
        "payload": {"graph_identity": "multi_scale_fusion"},
        "recipes": ["yolo26_generic_multi_scale_fusion"],
    },
    "feature_pyramid.multi_scale": {
        "adapter": "feature_pyramid.multi_scale",
        "changed": ["feature_pyramid.multi_scale.enabled"],
        "payload": {"graph_identity": "feature_pyramid_multi_scale"},
        "recipes": ["yolo26_feature_pyramid_multi_scale"],
    },
    "attention.spatial": {
        "adapter": "attention.spatial",
        "changed": ["attention.spatial.enabled"],
        "payload": {"graph_identity": "spatial_attention"},
        "recipes": ["yolo26_spatial_attention"],
    },
    "inference.sahi_slicing": {
        "adapter": "inference.sahi_slicing",
        "changed": ["inference.sahi_slicing.enabled"],
        "payload": {"inference_policy": "sahi_slicing", "training": False},
        "recipes": ["sahi_slicing_inference"],
    },
}


class PaperExecutionRequirementsBuilder:
    """Build requirements without collapsing papers by canonical component."""

    def __init__(self) -> None:
        # Import lazily: domain branch definitions validate against the paper
        # protocol module, which is re-exported by research.__init__.
        from yolo_agent.components.adapters.domain_adaptation.branches import (
            DomainAdaptationMethodRegistry,
        )

        self.distillation = DistillationMethodRegistry()
        self.domain = DomainAdaptationMethodRegistry()

    def build(
        self,
        inventory: PaperExecutionInventory,
        *,
        source_inventory_path: Path | str,
    ) -> PaperExecutionRequirementsMatrix:
        rows = [self._build_row(record) for record in inventory.records]
        return PaperExecutionRequirementsMatrix(
            source_inventory_path=str(source_inventory_path),
            source_inventory_hash=inventory.inventory_hash or inventory.calculate_hash(),
            compatible_paper_count=inventory.compatible_paper_count,
            requirements=rows,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _build_row(self, record: PaperExecutionSpec) -> PaperExecutionRequirement:
        mechanisms = set(record.canonical_component_ids)
        protocol = build_paper_protocol_contract(record.paper_id, mechanisms)
        if "domain_adaptation.general" in mechanisms:
            return self._domain_row(record, protocol)
        if "distillation.yolo26_teacher_student" in mechanisms:
            return self._distillation_row(record, protocol)
        if "inference.sahi_slicing" in mechanisms:
            return self._standard_row(record, protocol, "inference.sahi_slicing")
        specific = next(
            (item for item in record.paper_specific_mechanism_ids if item in _STANDARD_ROUTES),
            None,
        )
        if specific is None:
            specific = self._unresolved_mechanism(record)
            return self._unresolved_row(record, protocol, specific)
        return self._standard_row(record, protocol, specific)

    def _domain_row(
        self,
        record: PaperExecutionSpec,
        protocol: PaperProtocolContract,
    ) -> PaperExecutionRequirement:
        assignment = self.domain.assign(record.paper_id)
        branch = self.domain.get(assignment.branch_id)
        mechanism_ids = [assignment.branch_id]
        required_evidence = [
            *record.required_evidence,
            *protocol.required_evidence_artifacts,
        ]
        required_teacher_assets: list[str] = []
        if "distillation.yolo26_teacher_student" in record.canonical_component_ids:
            required_teacher_assets = [
                "frozen_teacher_checkpoint",
                "teacher_checkpoint_sha256",
                "teacher_student_same_split",
            ]
        blocker_codes = list(assignment.reason_codes) or ["domain_protocol_assets_missing"]
        blocker_codes.extend(
            [
                "source_dataset_manifest_sha256_required",
                "target_dataset_manifest_sha256_required",
                "explicit_domain_pair_required",
                "source_target_split_required",
                "label_availability_required",
            ]
        )
        disposition = (
            "evidence_recovery"
            if assignment.disposition == "evidence_recovery"
            else "implementation_request"
        )

        # A paper may require both domain adaptation and distillation. Preserve
        # both requirement branches instead of letting the first generic family
        # silently consume the second one.
        if "distillation.yolo26_teacher_student" in record.canonical_component_ids:
            distillation = self.distillation.assign(record.paper_id)
            if distillation.branch_id is None:
                mechanism_ids.append(self._unresolved_mechanism(record, family="distillation"))
                blocker_codes.extend(
                    [
                        "distillation_branch_unmapped",
                        "paper_method_identity_missing",
                    ]
                )
                required_evidence.extend(
                    [
                        "paper_specific_distillation_identity",
                        "teacher_checkpoint",
                        "teacher_checkpoint_sha256",
                        "teacher_student_same_split",
                    ]
                )
                disposition = "implementation_request"
            else:
                distillation_branch = self.distillation.get(distillation.branch_id)
                mechanism_ids.append(distillation.branch_id)
                required_evidence.extend(distillation_branch.evidence_schema)
                required_teacher_assets.extend(
                    [
                        "frozen_teacher_checkpoint",
                        "teacher_checkpoint_sha256",
                        "teacher_student_same_split",
                    ]
                )
                blocker_codes.extend(
                    [
                        "teacher_checkpoint_missing",
                        "teacher_checkpoint_sha256_missing",
                    ]
                )

        blocker = ";".join(dict.fromkeys(blocker_codes))
        required_domain_assets = [
            "source_domain_dataset" if branch.requires_source_domain else "source_trained_model_evidence",
            "target_domain_dataset",
            "explicit_source_target_domain_ids",
            "domain_pair_identity",
            "source_target_split",
            "label_availability",
        ]
        required_manifest_assets = ["target_domain_manifest"]
        if branch.requires_source_domain:
            required_manifest_assets.insert(0, "source_domain_manifest")
        required_evidence.extend(branch.required_evidence)
        return PaperExecutionRequirement(
            paper_id=record.paper_id,
            paper_specific_mechanism=assignment.branch_id,
            paper_specific_mechanism_ids=mechanism_ids,
            execution_route=disposition,
            required_adapter=branch.component_id,
            required_changed_variables=[branch.changed_variable],
            required_runtime_payload=dict(branch.payload_schema),
            required_evidence=list(dict.fromkeys(required_evidence)),
            required_dataset_protocol=protocol.model_dump(mode="json"),
            required_teacher_assets=required_teacher_assets,
            required_domain_assets=required_domain_assets,
            required_manifest_assets=required_manifest_assets,
            compatible_with_yolo26=True,
            training_candidate_allowed=False,
            exact_blocker=blocker,
            recovery_action=(
                "provide distinct hashed source/target manifests, explicit domain pair, "
                "split and label evidence; never use COCO train/val as domains"
            ),
            recipe_ids=[f"yolo26_{assignment.branch_id}"],
            current_disposition=disposition,
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=record.execution_fingerprint,
        )

    def _distillation_row(
        self,
        record: PaperExecutionSpec,
        protocol: PaperProtocolContract,
    ) -> PaperExecutionRequirement:
        assignment = self.distillation.assign(record.paper_id)
        if assignment.branch_id is None:
            return self._unresolved_row(
                record,
                protocol,
                self._unresolved_mechanism(record, family="distillation"),
                teacher=True,
            )
        branch = self.distillation.get(assignment.branch_id)
        recipe = f"yolo26_{assignment.branch_id.replace('_', '_')}"
        if assignment.branch_id == "teacher_ensemble":
            recipe = "yolo26_teacher_ensemble_distillation"
        elif assignment.branch_id == "cross_domain_teacher":
            recipe = "yolo26_cross_domain_teacher_distillation"
        elif assignment.branch_id == "contrastive_distillation":
            recipe = "yolo26_contrastive_distillation"
        elif assignment.branch_id == "source_free_teacher":
            recipe = "yolo26_source_free_teacher_distillation"
        ready_for_training = record.current_disposition == "runtime_ready"
        blocker = None if ready_for_training else (
            "teacher_checkpoint_missing;teacher_checkpoint_sha256_missing"
        )
        execution_route = "training" if ready_for_training else "blocked_runtime"
        return PaperExecutionRequirement(
            paper_id=record.paper_id,
            paper_specific_mechanism=assignment.branch_id,
            paper_specific_mechanism_ids=[assignment.branch_id],
            execution_route=execution_route,
            required_adapter=branch.component_id,
            required_changed_variables=sorted(branch.changed_variables),
            required_runtime_payload=dict(branch.runtime_payload_schema),
            required_evidence=[*record.required_evidence, *branch.evidence_schema],
            required_dataset_protocol=protocol.model_dump(mode="json"),
            required_teacher_assets=["frozen_teacher_checkpoint", "teacher_checkpoint_sha256", "teacher_student_same_split"],
            required_manifest_assets=["teacher_student_dataset_manifest"],
            compatible_with_yolo26=True,
            training_candidate_allowed=ready_for_training,
            exact_blocker=blocker,
            recovery_action="provide frozen teacher checkpoint, SHA-256, and matching teacher/student dataset manifests",
            recipe_ids=[recipe],
            current_disposition=record.current_disposition,
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=record.execution_fingerprint,
        )

    def _standard_row(
        self,
        record: PaperExecutionSpec,
        protocol: PaperProtocolContract,
        mechanism: str,
    ) -> PaperExecutionRequirement:
        spec = _STANDARD_ROUTES[mechanism]
        inference = mechanism.startswith("inference.")
        blocker = "inference_only_not_training_candidate" if inference else (
            None if record.current_disposition == "runtime_ready" else record.disposition_reason
        )
        route = "inference" if inference else ("training" if blocker is None else "blocked_runtime")
        return PaperExecutionRequirement(
            paper_id=record.paper_id,
            paper_specific_mechanism=mechanism,
            paper_specific_mechanism_ids=[mechanism],
            execution_route=route,
            required_adapter=str(spec["adapter"]),
            required_changed_variables=list(spec["changed"]),
            required_runtime_payload=dict(spec["payload"]),
            required_evidence=[*record.required_evidence, *protocol.required_evidence_artifacts],
            required_dataset_protocol=protocol.model_dump(mode="json"),
            required_graph_assets=["yolo26_one_to_one_head", "native_dfl_free_regression", "imgsz_640"]
            if protocol.is_model_graph else [],
            compatible_with_yolo26=True,
            training_candidate_allowed=route == "training",
            exact_blocker=blocker,
            recovery_action="complete runtime adapter contract, payload, evidence, and matched baseline before ASHA registration"
            if not inference else "run inference-only SAHI evaluation; never enqueue training ASHA",
            recipe_ids=list(spec["recipes"]),
            current_disposition=record.current_disposition,
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=record.execution_fingerprint,
        )

    def _unresolved_row(
        self,
        record: PaperExecutionSpec,
        protocol: PaperProtocolContract,
        mechanism: str,
        *,
        teacher: bool = False,
    ) -> PaperExecutionRequirement:
        return PaperExecutionRequirement(
            paper_id=record.paper_id,
            paper_specific_mechanism=mechanism,
            paper_specific_mechanism_ids=[mechanism],
            execution_route="implementation_request",
            required_adapter=None,
            required_changed_variables=[],
            required_runtime_payload={},
            required_evidence=[*record.required_evidence, "paper_specific_method_identity"],
            required_dataset_protocol=protocol.model_dump(mode="json"),
            required_teacher_assets=["frozen_teacher_checkpoint"] if teacher else [],
            required_domain_assets=["explicit_source_target_domain_assets"] if protocol.is_domain_adaptation else [],
            compatible_with_yolo26=True,
            training_candidate_allowed=False,
            exact_blocker="paper_specific_mechanism_unresolved",
            recovery_action="recover paper-specific method identity and bind a dedicated adapter/recipe before training",
            recipe_ids=list(record.recipe_ids),
            current_disposition="implementation_request",
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=record.execution_fingerprint,
        )

    @staticmethod
    def _unresolved_mechanism(
        record: PaperExecutionSpec,
        *,
        family: str | None = None,
    ) -> str:
        digest = hashlib.sha256(record.paper_id.encode()).hexdigest()[:12]
        unresolved_family = family or (
            "distillation"
            if any(
                item.startswith("distillation")
                for item in record.canonical_component_ids
            )
            else "paper"
        )
        return f"{unresolved_family}.unresolved_{digest}"


def build_paper_execution_requirements(
    inventory_path: Path | str = Path("runs/coverage-audit/paper_execution_inventory.yaml"),
    output_path: Path | str = Path("runs/coverage-audit/paper_execution_requirements.yaml"),
) -> PaperExecutionRequirementsMatrix:
    """Load inventory, build all rows, validate 83-paper coverage, and write YAML."""
    source = Path(inventory_path)
    inventory = PaperExecutionInventory.from_yaml(source)
    matrix = PaperExecutionRequirementsBuilder().build(inventory, source_inventory_path=source)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(matrix.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return matrix


__all__ = [
    "PaperExecutionRequirementsBuilder",
    "build_paper_execution_requirements",
]
