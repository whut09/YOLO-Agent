"""Generate the per-paper execution requirements matrix from the inventory."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from yolo_agent.components.adapters.domain_adaptation.domain_paper_routes import (
        DomainPaperRouteRegistry,
    )
    from yolo_agent.components.adapters.distillation.paper_routes import (
        DistillationPaperRouteRegistry,
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
from yolo_agent.resources import ResourcePaths


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


def validate_independent_standard_routes() -> list[str]:
    """Return readiness-binding errors for the independent component routes.

    Every independent component must keep a complete standard route:
    adapter, changed variables, runtime payload, and the router's bound
    recipe.  A readiness matrix can never silently drop one of these
    fields.
    """
    from yolo_agent.components.independent_component_router import (
        COMPONENT_CATALOG,
        INDEPENDENT_COMPONENT_IDS,
    )

    errors: list[str] = []
    for component_id in INDEPENDENT_COMPONENT_IDS:
        spec = _STANDARD_ROUTES.get(component_id)
        if spec is None:
            errors.append(f"independent_route_missing:{component_id}")
            continue
        if not spec.get("adapter"):
            errors.append(f"independent_route_adapter_missing:{component_id}")
        if not spec.get("changed"):
            errors.append(f"independent_route_changed_missing:{component_id}")
        if not spec.get("payload"):
            errors.append(f"independent_route_payload_missing:{component_id}")
        if not spec.get("recipes"):
            errors.append(f"independent_route_recipes_missing:{component_id}")
        expected_recipe = str(COMPONENT_CATALOG[component_id]["recipe_id"])
        if expected_recipe not in spec.get("recipes", []):
            errors.append(
                f"independent_route_recipe_mismatch:{component_id}:{expected_recipe}"
            )
    return errors


class PaperExecutionRequirementsBuilder:
    """Build requirements without collapsing papers by canonical component."""

    def __init__(
        self,
        paper_routes: DistillationPaperRouteRegistry | None = None,
        domain_routes: DomainPaperRouteRegistry | None = None,
    ) -> None:
        # Import lazily: domain branch definitions validate against the paper
        # protocol module, which is re-exported by research.__init__.
        from yolo_agent.components.adapters.domain_adaptation.branches import (
            DomainAdaptationMethodRegistry,
        )
        from yolo_agent.components.adapters.domain_adaptation.domain_paper_routes import (
            default_domain_paper_route_registry,
        )
        from yolo_agent.components.adapters.distillation.method_registry import (
            DistillationMethodRegistry,
        )
        from yolo_agent.components.adapters.distillation.paper_routes import (
            default_paper_route_registry,
        )

        self.distillation = DistillationMethodRegistry()
        self.paper_routes = (
            paper_routes
            if paper_routes is not None
            else default_paper_route_registry()
        )
        self.domain = DomainAdaptationMethodRegistry()
        self.domain_routes = (
            domain_routes
            if domain_routes is not None
            else default_domain_paper_route_registry()
        )
        # Paper-specific recipes are an additional identity source for papers
        # whose catalog profile still contains a generic component.  Loading
        # this index here keeps the requirements matrix aligned with the
        # recipe boundary without changing the frozen 83-paper denominator.
        self._paper_recipe_routes = _load_paper_recipe_routes()
        self._contracts = _load_component_contracts()

    def build(
        self,
        inventory: PaperExecutionInventory,
        *,
        source_inventory_path: Path | str,
    ) -> PaperExecutionRequirementsMatrix:
        independent_errors = validate_independent_standard_routes()
        if independent_errors:
            raise ValueError(
                "independent route readiness binding incomplete: "
                + ", ".join(independent_errors)
            )
        rows = [self._build_row(record) for record in inventory.records]
        return PaperExecutionRequirementsMatrix(
            source_inventory_path=str(source_inventory_path),
            source_inventory_hash=inventory.inventory_hash or inventory.calculate_hash(),
            compatible_paper_count=inventory.compatible_paper_count,
            requirements=rows,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _build_row(self, record: PaperExecutionSpec) -> PaperExecutionRequirement:
        mechanisms = sorted(set(record.canonical_component_ids))
        protocol = build_paper_protocol_contract(record.paper_id, mechanisms)
        if "domain_adaptation.general" in mechanisms:
            return self._domain_row(record, protocol)
        if "distillation.yolo26_teacher_student" in mechanisms:
            return self._distillation_row(record, protocol)
        if "inference.sahi_slicing" in mechanisms:
            return self._standard_row(record, protocol, "inference.sahi_slicing")
        paper_recipe = self._paper_recipe_routes.get(record.paper_id)
        if paper_recipe is not None:
            return self._paper_recipe_row(record, protocol, paper_recipe)
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
        route = self.domain_routes.route(record.paper_id)
        assignment = self.domain.assign(record.paper_id)
        branch = self.domain.get(assignment.branch_id)
        mechanism_ids = [route.paper_specific_mechanism_id, assignment.branch_id]
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
            paper_specific_mechanism=route.paper_specific_mechanism_id,
            paper_specific_mechanism_ids=mechanism_ids,
            execution_route=disposition,
            required_adapter=route.adapter_class,
            required_changed_variables=sorted(
                set(route.changed_variables) | {route.branch_changed_variable}
            ),
            required_runtime_payload=dict(route.route_payload_schema),
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
            recipe_ids=[route.recipe_id],
            current_disposition=disposition,
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=route.execution_fingerprint
            or record.execution_fingerprint,
        )

    def _distillation_row(
        self,
        record: PaperExecutionSpec,
        protocol: PaperProtocolContract,
    ) -> PaperExecutionRequirement:
        route = self.paper_routes.route(record.paper_id)
        assignment = self.distillation.assign(record.paper_id)
        branch = (
            self.distillation.get(assignment.branch_id)
            if assignment.branch_id is not None
            else None
        )
        ready_for_training = record.current_disposition == "runtime_ready"
        if branch is None:
            # The paper owns an identity-recovery route: its mechanism,
            # adapter class, recipe, changed variables, and payload schema are
            # concrete even though the branch method itself is still unmapped.
            # Keep the identity-recovery blocker; it must not erase the
            # adapter route or turn a paper into a generic candidate.
            mechanism_ids = [route.paper_specific_mechanism_id]
            required_evidence = [
                *record.required_evidence,
                "paper_specific_distillation_identity",
            ]
            execution_route = "implementation_request"
            required_adapter = route.adapter_class
            changed_variables = sorted(route.changed_variables)
            blocker = ";".join(
                [
                    *route.reason_codes,
                    "teacher_checkpoint_missing",
                    "teacher_checkpoint_sha256_missing",
                ]
            )
            recovery_action = (
                "recover the paper-specific method identity and provide a "
                "frozen teacher checkpoint, SHA-256, and matching "
                "teacher/student dataset manifests"
            )
        else:
            mechanism_ids = [
                route.paper_specific_mechanism_id,
                assignment.branch_id,
            ]
            required_evidence = [*record.required_evidence, *branch.evidence_schema]
            execution_route = "training" if ready_for_training else "blocked_runtime"
            required_adapter = route.adapter_class
            changed_variables = sorted(route.changed_variables)
            blocker = None if ready_for_training else (
                "teacher_checkpoint_missing;teacher_checkpoint_sha256_missing"
            )
            recovery_action = (
                "provide frozen teacher checkpoint, SHA-256, and matching "
                "teacher/student dataset manifests"
            )
        return PaperExecutionRequirement(
            paper_id=record.paper_id,
            paper_specific_mechanism=route.paper_specific_mechanism_id,
            paper_specific_mechanism_ids=mechanism_ids,
            execution_route=execution_route,
            required_adapter=required_adapter,
            required_changed_variables=changed_variables,
            required_runtime_payload=dict(route.runtime_payload_schema),
            required_evidence=list(dict.fromkeys(required_evidence)),
            required_dataset_protocol=protocol.model_dump(mode="json"),
            required_teacher_assets=["frozen_teacher_checkpoint", "teacher_checkpoint_sha256", "teacher_student_same_split"],
            required_manifest_assets=["teacher_student_dataset_manifest"],
            compatible_with_yolo26=True,
            training_candidate_allowed=ready_for_training,
            exact_blocker=blocker,
            recovery_action=recovery_action,
            recipe_ids=[route.recipe_id],
            current_disposition=record.current_disposition,
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=route.execution_fingerprint
            or record.execution_fingerprint,
        )

    def _paper_recipe_row(
        self,
        record: PaperExecutionSpec,
        protocol: PaperProtocolContract,
        recipe: dict[str, Any],
    ) -> PaperExecutionRequirement:
        """Bind a generic profile to its explicit paper recipe.

        The recipe is deliberately not treated as readiness evidence.  It
        supplies the missing paper identity and adapter route; CPU maturity,
        runtime evidence, and matched-control checks remain separate gates.
        """
        mechanism = str(recipe.get("paper_specific_mechanism_id") or "").strip()
        component_ids = [str(item) for item in recipe.get("component_ids", [])]
        component_id = component_ids[0] if component_ids else mechanism
        contract = self._contracts.get(component_id) or self._contracts.get(mechanism)
        adapter = str(contract.adapter_class) if contract and contract.adapter_class else None
        changed = []
        if contract and contract.changed_variable:
            changed.append(str(contract.changed_variable))
        primary = str(recipe.get("primary_changed_variable") or "").strip()
        if primary:
            changed.append(primary)
        changed = sorted(set(changed))
        payload = dict(contract.runtime_payload_schema) if contract else {}
        payload.update(
            {
                "paper_id": "str",
                "paper_specific_mechanism_id": "str",
                "recipe_id": "str",
            }
        )
        evidence = [
            *record.required_evidence,
            *protocol.required_evidence_artifacts,
        ]
        if contract:
            evidence.extend(contract.evidence_protocol)
        evidence.extend(
            [
                "paper_specific_method_identity",
                "recipe_contract_and_runtime_payload",
            ]
        )
        reasons: list[str] = []
        if not mechanism:
            reasons.append("paper_recipe_mechanism_missing")
        if not adapter:
            reasons.append(f"paper_specific_adapter_contract_missing:{component_id}")
        if not changed:
            reasons.append(f"paper_specific_changed_variable_missing:{component_id}")
        if not contract or not contract.can_execute:
            reasons.append("paper_specific_adapter_runtime_evidence_missing")
        if record.current_disposition != "runtime_ready":
            reasons.append("paper_profile_runtime_evidence_incomplete")
        if not reasons:
            execution_route = "training"
            training_allowed = True
            blocker = None
        else:
            execution_route = "implementation_request"
            training_allowed = False
            blocker = ";".join(dict.fromkeys(reasons))
        return PaperExecutionRequirement(
            paper_id=record.paper_id,
            paper_specific_mechanism=mechanism,
            paper_specific_mechanism_ids=[mechanism],
            execution_route=execution_route,
            required_adapter=adapter,
            required_changed_variables=changed,
            required_runtime_payload=payload,
            required_evidence=sorted(set(evidence)),
            required_dataset_protocol=protocol.model_dump(mode="json"),
            required_graph_assets=[
                "yolo26_one_to_one_head",
                "native_dfl_free_regression",
                "imgsz_640",
            ] if contract and contract.changes_model_graph is True else [],
            compatible_with_yolo26=(
                bool(contract and contract.fixed_imgsz_compatible is not False)
            ),
            training_candidate_allowed=training_allowed,
            exact_blocker=blocker,
            recovery_action=(
                "run the paper-specific CPU contract/forward/backward checks, "
                "persist non-mock runtime evidence, and regenerate the "
                "requirements matrix before ASHA registration"
            ),
            recipe_ids=[str(recipe["recipe_id"])],
            current_disposition=(
                "runtime_ready" if training_allowed else "implementation_request"
            ),
            protocol_hash=protocol.protocol_hash,
            execution_fingerprint=_paper_recipe_fingerprint(
                record,
                protocol=protocol,
                recipe=recipe,
                component_id=component_id,
                contract=contract,
                adapter=adapter,
                changed_variables=changed,
                payload=payload,
            ),
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


def _load_paper_recipe_routes() -> dict[str, dict[str, Any]]:
    """Load explicit paper recipe identities without changing the denominator."""
    path = ResourcePaths.RECIPES_DIR / "paper_specific_methods.yaml"
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    entries = raw.get("recipes", []) if isinstance(raw, dict) else []
    routes: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mechanism = str(entry.get("paper_specific_mechanism_id") or "").strip()
        recipe_id = str(entry.get("recipe_id") or "").strip()
        if not mechanism or not recipe_id:
            continue
        for paper_id in entry.get("paper_ids", []):
            paper_key = str(paper_id).strip()
            if not paper_key:
                continue
            previous = routes.get(paper_key)
            if previous is not None and (
                previous.get("recipe_id") != recipe_id
                or previous.get("paper_specific_mechanism_id") != mechanism
            ):
                raise ValueError(
                    "paper has conflicting explicit recipe routes: " + paper_key
                )
            routes[paper_key] = entry
    return routes


def _load_component_contracts() -> dict[str, Any]:
    """Return the highest available local contract for route binding."""
    from yolo_agent.research.component_aliases import ComponentAliasResolver

    return dict(ComponentAliasResolver.from_yaml().contracts)


def _paper_recipe_fingerprint(
    record: PaperExecutionSpec,
    *,
    protocol: PaperProtocolContract,
    recipe: dict[str, Any],
    component_id: str,
    contract: Any | None,
    adapter: str | None,
    changed_variables: list[str],
    payload: dict[str, Any],
) -> str:
    """Hash the explicit paper route instead of its generic catalog profile."""
    identity = {
        "paper_id": record.paper_id,
        "profile_id": record.profile_id,
        "component_id": component_id,
        "paper_specific_mechanism_id": recipe.get("paper_specific_mechanism_id"),
        "recipe_id": recipe.get("recipe_id"),
        "recipe_version": recipe.get("version", ""),
        "adapter": adapter,
        "implementation_path": (
            str(contract.implementation_path) if contract is not None else None
        ),
        "changed_variables": changed_variables,
        "runtime_payload_schema": payload,
        "component_contract_signature": (
            contract.runtime_contract_signature if contract is not None else None
        ),
        "protocol_hash": protocol.protocol_hash,
        "catalog_fingerprint": record.execution_fingerprint,
    }
    encoded = yaml.safe_dump(identity, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "PaperExecutionRequirementsBuilder",
    "build_paper_execution_requirements",
    "validate_independent_standard_routes",
]
