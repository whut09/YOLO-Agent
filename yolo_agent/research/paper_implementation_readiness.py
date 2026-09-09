"""Evaluate implementation readiness for each frozen paper independently.

This module is deliberately an offline evaluator.  It consumes existing paper
profiles, execution inventory rows, component contracts, and file-backed
maturity artifacts.  It never creates evidence, runs a trainer, or changes the
component maturity registry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from yolo_agent.components.adapters.base import ComponentAdapter
from yolo_agent.components.adapters.runtime import AdapterRuntimePayload
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.components.maturity import ComponentMaturityArtifact, maturity_rank
from yolo_agent.research.executable_coverage_schemas import PaperExecutableCoverageEntry
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodProfile,
)
from yolo_agent.research.paper_83_campaign_schemas import Paper83Paper
from yolo_agent.research.paper_execution_schemas import PaperExecutionSpec
from yolo_agent.research.paper_mechanism_resolver import GENERIC_MECHANISM_IDS
from yolo_agent.research.paper_implementation_schemas import (
    ImplementationEvidenceClass,
    PaperImplementationReadiness,
    PaperImplementationSpec,
)


_GENERIC_METHOD_TERMS = frozenset(
    {
        "knowledge_distillation",
        "distillation",
        "domain_adaptation",
        "quality_alignment",
        "quality_alignment_loss",
        "incremental_detection",
        "few_shot_detection",
        "open_vocabulary",
        "aerial_detection",
        "semi_supervised",
        "small_object",
    }
)
_COMPATIBILITY_TERMS = ("compatib", "yolo26", "one_to_one", "dfl_free")
_SMOKE_TERMS = ("smoke", "forward", "backward", "synthetic", "shape")


def _unique(values: Iterable[Any]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _is_generic_mechanism(value: str) -> bool:
    normalized = value.strip().casefold().replace("-", "_")
    return normalized in _GENERIC_METHOD_TERMS or value in GENERIC_MECHANISM_IDS or (
        normalized.endswith(".general")
    )


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PaperImplementationReadinessEvaluator:
    """Build one strict paper implementation spec without training."""

    def __init__(
        self,
        *,
        manifest_membership_hash: str,
        contracts: Mapping[str, ComponentContract] | Iterable[ComponentContract] = (),
        tests_root: Path | str | None = Path("tests"),
    ) -> None:
        if len(manifest_membership_hash) != 64:
            raise ValueError("paper readiness requires a manifest membership hash")
        if isinstance(contracts, Mapping):
            self.contracts = dict(contracts)
        else:
            self.contracts = {item.component_id: item for item in contracts}
        self.manifest_membership_hash = manifest_membership_hash
        self.tests_root = Path(tests_root).resolve() if tests_root is not None else None
        self._test_cache: dict[
            tuple[str, str], tuple[list[str], list[str], list[str]]
        ] = {}

    def evaluate(
        self,
        *,
        paper: Paper83Paper,
        profile: PaperMethodProfile | None,
        decision: PaperImplementationDecision | None = None,
        inventory: PaperExecutionSpec | None = None,
        coverage: PaperExecutableCoverageEntry | None = None,
    ) -> PaperImplementationSpec:
        """Evaluate one paper and retain all blockers on that paper."""

        if decision is not None and decision.paper_id != paper.paper_id:
            raise ValueError("paper decision does not match frozen paper ID")
        if inventory is not None and inventory.paper_id != paper.paper_id:
            raise ValueError("paper execution inventory row does not match frozen paper ID")
        if coverage is not None and coverage.paper_id != paper.paper_id:
            raise ValueError("paper coverage row does not match frozen paper ID")

        blockers: list[str] = []
        profile_id = paper.method_profile_id
        source_locations = list(profile.source_locations) if profile else []
        if profile is None:
            blockers.append(f"missing_method_profile:{paper.paper_id}")
        else:
            profile_id = profile.profile_id
            if profile.paper_id != paper.paper_id:
                blockers.append("method_profile_paper_id_mismatch")
            if profile.profile_id != paper.method_profile_id:
                blockers.append("method_profile_id_changed_from_manifest")
            source_locations = _unique(source_locations)

        if decision is None:
            blockers.append("missing_implementation_decision")
        elif decision.profile_id != profile_id:
            blockers.append("implementation_decision_profile_id_mismatch")

        specific_mechanisms, mechanism_sources = self._paper_specific_mechanisms(
            profile=profile,
            decision=decision,
            inventory=inventory,
        )
        component_ids = self._component_ids(
            paper=paper,
            profile=profile,
            decision=decision,
            inventory=inventory,
            coverage=coverage,
        )
        generic_components = {
            item for item in component_ids if _is_generic_mechanism(item)
        }
        shared_primitives = _unique(
            [item for item in component_ids if item not in set(mechanism_sources)]
            + [item for item in generic_components]
        )
        adapter_ids = self._adapter_ids(
            paper=paper,
            component_ids=component_ids,
            profile=profile,
            decision=decision,
            inventory=inventory,
            coverage=coverage,
        )

        insertion_points = self._insertion_points(
            profile=profile,
            coverage=coverage,
            component_ids=component_ids,
        )
        runtime_hooks, payloads, payload_blockers = self._runtime_hooks(
            profile=profile,
            coverage=coverage,
            component_ids=component_ids,
        )
        blockers.extend(payload_blockers)

        changes = self._change_buckets(
            profile=profile,
            component_ids=component_ids,
            insertion_points=insertion_points,
        )
        paper_config = self._paper_config(
            profile=profile,
            decision=decision,
            inventory=inventory,
            specific_mechanisms=specific_mechanisms,
            mechanism_sources=mechanism_sources,
            insertion_points=insertion_points,
            runtime_hooks=runtime_hooks,
            payloads=payloads,
        )
        if not specific_mechanisms:
            evidence_class: ImplementationEvidenceClass = (
                "generic_only" if generic_components else "metadata_only"
            )
            blockers.append(
                "generic_only:paper_specific_mechanism_missing"
                if generic_components
                else "unknown_mechanism:paper_specific_mechanism_missing"
            )
        else:
            evidence_class = "paper_specific"
            if not paper_config:
                blockers.append("missing_paper_config")

        if not component_ids:
            blockers.append("missing_component_binding")
        if not adapter_ids:
            blockers.append("missing_adapter_binding")
        if not insertion_points:
            blockers.append("missing_runtime_insertion_point")
        if not runtime_hooks:
            blockers.append("missing_runtime_hook")

        runtime_verified = True
        unit_passed = True
        smoke_passed = True
        compatibility_passed = True
        unit_refs: list[str] = []
        smoke_refs: list[str] = []
        compatibility_refs: list[str] = []

        for component_id in component_ids:
            contract = self.contracts.get(component_id)
            if contract is None:
                runtime_verified = False
                unit_passed = False
                smoke_passed = False
                compatibility_passed = False
                blockers.append(f"missing_adapter_contract:{component_id}")
                continue

            binding_error = self._adapter_binding_error(contract)
            if binding_error:
                runtime_verified = False
                blockers.append(f"{binding_error}:{component_id}")

            stage_ok, stage_reason = self._runtime_stage(contract, payloads.get(component_id))
            if not stage_ok:
                runtime_verified = False
                blockers.append(f"{stage_reason}:{component_id}")

            unit_artifact_ok = self._artifact_ok(contract, "unit_tested")
            smoke_artifact_ok = self._artifact_ok(contract, "smoke_passed")
            if not unit_artifact_ok:
                unit_passed = False
                blockers.append(f"component_unit_evidence_missing:{component_id}")
            if not smoke_artifact_ok:
                smoke_passed = False
                blockers.append(f"component_smoke_evidence_missing:{component_id}")

            refs = self._test_refs(component_id, contract.adapter_class or "")
            unit_refs.extend(refs[0])
            smoke_refs.extend(refs[1])
            compatibility_refs.extend(refs[2])
            if not refs[0]:
                unit_passed = False
                blockers.append(f"missing_test:unit:{component_id}")
            if not refs[1]:
                smoke_passed = False
                blockers.append(f"missing_test:smoke:{component_id}")
            if not refs[2]:
                compatibility_passed = False
                blockers.append(f"missing_test:compatibility:{component_id}")

            compatibility_errors = self._compatibility_errors(contract)
            if compatibility_errors:
                compatibility_passed = False
                blockers.extend(
                    f"{reason}:{component_id}" for reason in compatibility_errors
                )

        if not component_ids:
            runtime_verified = False
            unit_passed = False
            smoke_passed = False
            compatibility_passed = False

        paper_evidence_refs = self._paper_evidence_refs(profile)
        if not paper_evidence_refs:
            blockers.append("paper_specific_evidence_unbound")
        official_code_refs, source_license = self._official_code(profile)
        required_assets = self._required_assets(profile, inventory)
        hyperparameters = self._hyperparameters(profile)
        fingerprint = _sha256_payload(
            {
                "manifest_membership_hash": self.manifest_membership_hash,
                "paper_id": paper.paper_id,
                "profile_id": profile_id,
                "paper_specific_mechanisms": specific_mechanisms,
                "component_ids": component_ids,
                "adapter_ids": adapter_ids,
                "paper_specific_config": paper_config,
                "paper_specific_hyperparameters": hyperparameters,
                "runtime_insertion_points": insertion_points,
                "runtime_hooks": runtime_hooks,
                "contract_signatures": {
                    component_id: self.contracts[component_id].runtime_contract_signature
                    for component_id in component_ids
                    if component_id in self.contracts
                },
            }
        )

        if evidence_class in {"generic_only", "metadata_only", "recipe_only", "unknown"}:
            blockers.append(f"{evidence_class}:paper_implementation_not_specific")
        blockers = _unique(blockers)
        readiness = self._readiness_level(
            profile=profile,
            specific_mechanisms=specific_mechanisms,
            paper_config=paper_config,
            component_ids=component_ids,
            adapter_ids=adapter_ids,
            insertion_points=insertion_points,
            runtime_hooks=runtime_hooks,
            runtime_verified=runtime_verified,
            unit_passed=unit_passed,
            smoke_passed=smoke_passed,
            compatibility_passed=compatibility_passed,
            evidence_class=evidence_class,
            blockers=blockers,
        )

        implementation_domain = self._implementation_domain(profile)
        mechanism_summary = self._mechanism_summary(profile, specific_mechanisms)
        return PaperImplementationSpec(
            paper_id=paper.paper_id,
            manifest_membership_hash=self.manifest_membership_hash,
            method_profile_id=profile_id,
            title=paper.title,
            year=paper.year,
            implementation_domain=implementation_domain,
            mechanism_summary=mechanism_summary,
            paper_specific_mechanisms=specific_mechanisms,
            shared_primitives=shared_primitives,
            component_ids=component_ids,
            adapter_ids=adapter_ids,
            runtime_insertion_points=insertion_points,
            runtime_hooks=runtime_hooks,
            architecture_changes=changes["architecture"],
            data_changes=changes["data"],
            annotation_changes=changes["annotation"],
            sampling_changes=changes["sampling"],
            augmentation_changes=changes["augmentation"],
            loss_changes=changes["loss"],
            assignment_changes=changes["assignment"],
            training_changes=changes["training"],
            postprocess_changes=changes["postprocess"],
            inference_changes=changes["inference"],
            paper_specific_config=paper_config,
            paper_specific_hyperparameters=hyperparameters,
            required_assets=required_assets,
            paper_evidence_refs=paper_evidence_refs,
            official_code_refs=official_code_refs,
            source_license=source_license,
            unit_test_refs=_unique(unit_refs),
            smoke_test_refs=_unique(smoke_refs),
            compatibility_test_refs=_unique(compatibility_refs),
            implementation_fingerprint=fingerprint,
            readiness=readiness,
            blockers=blockers,
            implementation_evidence_class=evidence_class,
            runtime_implementation_verified=runtime_verified,
            unit_tests_passed=unit_passed,
            non_mock_smoke_passed=smoke_passed,
            compatibility_validation_passed=compatibility_passed,
            pilot_reproduced=False,
            full_reproduced=False,
            confirmed_multi_seed=False,
        )

    def _paper_specific_mechanisms(
        self,
        *,
        profile: PaperMethodProfile | None,
        decision: PaperImplementationDecision | None,
        inventory: PaperExecutionSpec | None,
    ) -> tuple[list[str], set[str]]:
        """Collect only explicitly authorized paper-level mechanisms.

        ``paper_component_ids`` and ``method_name`` are catalog metadata.  They
        are useful for explaining a gap, but they are not sufficient evidence
        that a generic adapter implements the source paper.  The inventory and
        resolver outputs are the canonical boundaries; structured observations
        and decision mappings must carry the same explicit authorization.
        """
        values: set[str] = set()
        sources: set[str] = set()
        if inventory is not None:
            for item in inventory.paper_specific_mechanism_ids:
                if not _is_generic_mechanism(item):
                    values.add(item)
                    sources.add(item)
            for resolution in inventory.paper_mechanism_resolutions:
                if (
                    resolution.resolved
                    and resolution.paper_specific_mechanism_id
                    and not _is_generic_mechanism(
                        resolution.paper_specific_mechanism_id
                    )
                    and resolution.canonical_component_id
                    and not _is_generic_mechanism(
                        resolution.canonical_component_id
                    )
                ):
                    values.add(resolution.paper_specific_mechanism_id)
                    sources.add(resolution.canonical_component_id)
        if profile is not None:
            for resolution in profile.paper_mechanism_resolutions:
                if (
                    resolution.resolved
                    and resolution.paper_specific_mechanism_id
                    and not _is_generic_mechanism(
                        resolution.paper_specific_mechanism_id
                    )
                    and resolution.canonical_component_id
                    and not _is_generic_mechanism(
                        resolution.canonical_component_id
                    )
                ):
                    values.add(resolution.paper_specific_mechanism_id)
                    sources.add(resolution.canonical_component_id or "")
            structured = profile.structured_method_evidence
            if structured is not None and structured.authorizes_method_profile:
                authorized_mechanisms = {
                    str(item.value)
                    for item in structured.observations
                    if item.field_name == "canonical_mechanism"
                    and item.authorizes_method_profile
                    and isinstance(item.value, str)
                }
                for value in structured.canonical_mechanisms:
                    if value in authorized_mechanisms and not _is_generic_mechanism(
                        value
                    ):
                        values.add(value)
                        sources.add(value)
        if decision is not None:
            for mapping in decision.mechanism_mappings:
                if (
                    mapping.authorizes_method_profile
                    and not _is_generic_mechanism(mapping.source_term)
                    and not _is_generic_mechanism(mapping.canonical_component_id)
                ):
                    values.add(mapping.source_term)
                    sources.add(mapping.canonical_component_id)
        return _unique(values), {item for item in sources if item}

    def _component_ids(
        self,
        *,
        paper: Paper83Paper,
        profile: PaperMethodProfile | None,
        decision: PaperImplementationDecision | None,
        inventory: PaperExecutionSpec | None,
        coverage: PaperExecutableCoverageEntry | None,
    ) -> list[str]:
        values: list[str] = list(paper.current_component_ids)
        if profile is not None:
            values.extend(profile.canonical_component_ids)
        if decision is not None:
            values.extend(decision.canonical_component_ids)
        if inventory is not None:
            values.extend(inventory.canonical_component_ids)
        if coverage is not None:
            values.extend(coverage.canonical_mechanisms)
        return _unique(values)

    def _adapter_ids(
        self,
        *,
        paper: Paper83Paper,
        component_ids: list[str],
        profile: PaperMethodProfile | None,
        decision: PaperImplementationDecision | None,
        inventory: PaperExecutionSpec | None,
        coverage: PaperExecutableCoverageEntry | None,
    ) -> list[str]:
        values: list[str] = list(paper.current_adapter_ids)
        values.extend(paper.frozen_certified_adapter_ids)
        if decision is not None:
            values.extend(decision.reusable_adapter_ids)
            values.extend(decision.required_adapter_ids)
        if inventory is not None:
            values.extend(inventory.reusable_adapter_ids)
            values.extend(inventory.runtime_ready_adapters)
            values.extend(
                item.required_adapter
                for item in inventory.paper_mechanism_resolutions
                if item.required_adapter
            )
        if coverage is not None:
            values.extend(coverage.reusable_adapter_candidates)
            values.extend(coverage.runtime_ready_adapters)
        values = [item for item in values if item in component_ids or item in self.contracts]
        return _unique(values)

    def _insertion_points(
        self,
        *,
        profile: PaperMethodProfile | None,
        coverage: PaperExecutableCoverageEntry | None,
        component_ids: list[str],
    ) -> list[str]:
        values: list[str] = []
        if profile is not None:
            values.extend(profile.protocol_constraints.get("insertion_points", []))
        if coverage is not None:
            values.extend(coverage.required_runtime_hooks)
        values.extend(
            contract.insertion_point
            for component_id in component_ids
            if (contract := self.contracts.get(component_id)) is not None
            and contract.insertion_point != "unknown"
        )
        return _unique(values)

    def _runtime_hooks(
        self,
        *,
        profile: PaperMethodProfile | None,
        coverage: PaperExecutableCoverageEntry | None,
        component_ids: list[str],
    ) -> tuple[list[str], dict[str, AdapterRuntimePayload], list[str]]:
        values: list[str] = []
        blockers: list[str] = []
        if profile is not None:
            values.extend(profile.protocol_constraints.get("required_runtime_hooks", []))
            structured = profile.structured_method_evidence
            if structured is not None:
                values.extend(structured.required_runtime_hooks)
        if coverage is not None:
            values.extend(coverage.required_runtime_hooks)
        payloads: dict[str, AdapterRuntimePayload] = {}
        for component_id in component_ids:
            contract = self.contracts.get(component_id)
            if contract is None:
                continue
            artifact = self._latest_artifact(contract, "runtime_integrated")
            if artifact is None:
                continue
            try:
                payload = AdapterRuntimePayload.read(artifact.artifact_path, verify_imports=True)
            except (OSError, TypeError, ValueError, ImportError) as exc:
                blockers.append(f"runtime_payload_invalid:{component_id}:{type(exc).__name__}")
                continue
            payloads[component_id] = payload
            values.extend(
                hook
                for plugin in payload.plugin_references
                for hook in plugin.required_hooks
            )
        return _unique(values), dict(payloads), blockers

    def _paper_config(
        self,
        *,
        profile: PaperMethodProfile | None,
        decision: PaperImplementationDecision | None,
        inventory: PaperExecutionSpec | None,
        specific_mechanisms: list[str],
        mechanism_sources: set[str],
        insertion_points: list[str],
        runtime_hooks: list[str],
        payloads: Mapping[str, AdapterRuntimePayload],
    ) -> dict[str, Any]:
        if not specific_mechanisms:
            return {}
        parameters = dict(profile.paper_parameters) if profile is not None else {}
        protocol = dict(profile.protocol_constraints) if profile is not None else {}
        changed = parameters.get("changed_variables", [])
        resolution_config: list[dict[str, Any]] = []
        if profile is not None:
            resolution_config.extend(
                item.model_dump(mode="json")
                for item in profile.paper_mechanism_resolutions
                if item.paper_specific_mechanism_id in specific_mechanisms
            )
        if inventory is not None:
            resolution_config.extend(
                item.model_dump(mode="json")
                for item in inventory.paper_mechanism_resolutions
                if item.paper_specific_mechanism_id in specific_mechanisms
            )
        has_paper_inputs = bool(
            changed
            or protocol
            or insertion_points
            or runtime_hooks
            or resolution_config
        )
        if not has_paper_inputs:
            return {}
        return {
            "paper_specific_mechanisms": list(specific_mechanisms),
            "source_mechanism_ids": _unique(mechanism_sources),
            "changed_variables": changed,
            "insertion_points": insertion_points,
            "required_runtime_hooks": runtime_hooks,
            "protocol_constraints": protocol,
            "mechanism_resolutions": resolution_config,
            "runtime_payload_component_ids": sorted(payloads),
        }

    def _change_buckets(
        self,
        *,
        profile: PaperMethodProfile | None,
        component_ids: list[str],
        insertion_points: list[str],
    ) -> dict[str, list[str]]:
        parameters = profile.paper_parameters if profile is not None else {}
        protocol = profile.protocol_constraints if profile is not None else {}
        changed = _unique(parameters.get("changed_variables", []))
        method_families = _unique(protocol.get("method_families", []))
        buckets = {name: [] for name in (
            "architecture",
            "data",
            "annotation",
            "sampling",
            "augmentation",
            "loss",
            "assignment",
            "training",
            "postprocess",
            "inference",
        )}
        for value in changed + method_families:
            lower = value.casefold()
            if any(token in lower for token in ("backbone", "neck", "head", "graph", "feature")):
                buckets["architecture"].append(value)
            elif "annotation" in lower or "label" in lower:
                buckets["annotation"].append(value)
            elif "sample" in lower or "replay" in lower:
                buckets["sampling"].append(value)
            elif "augment" in lower or "crop" in lower or "mix" in lower:
                buckets["augmentation"].append(value)
            elif "assign" in lower or "matching" in lower:
                buckets["assignment"].append(value)
            elif "loss" in lower or "distill" in lower or "quality" in lower:
                buckets["loss"].append(value)
            elif "infer" in lower or "sahi" in lower or "slice" in lower:
                buckets["inference"].append(value)
            elif "post" in lower or "nms" in lower:
                buckets["postprocess"].append(value)
            elif "data" in lower or "dataset" in lower or "domain" in lower:
                buckets["data"].append(value)
            else:
                buckets["training"].append(value)
        buckets["architecture"].extend(insertion_points)
        for component_id in component_ids:
            contract = self.contracts.get(component_id)
            if contract is None:
                continue
            if contract.changes_model_graph is True:
                buckets["architecture"].append(component_id)
            if contract.training_only is True:
                buckets["training"].append(component_id)
            if contract.inference_only is True:
                buckets["inference"].append(component_id)
        return {key: _unique(value) for key, value in buckets.items()}

    def _runtime_stage(
        self,
        contract: ComponentContract,
        payload: AdapterRuntimePayload | None,
    ) -> tuple[bool, str]:
        if maturity_rank(contract.maturity) < maturity_rank("runtime_integrated"):
            return False, "component_runtime_evidence_missing"
        if payload is None:
            return False, "runtime_payload_missing"
        if contract.component_id not in payload.component_ids:
            return False, "runtime_payload_component_mismatch"
        return True, "ok"

    def _adapter_binding_error(self, contract: ComponentContract) -> str | None:
        if not contract.implementation_path or not contract.adapter_class:
            return "missing_runtime:implementation_identity"
        try:
            module = importlib.import_module(contract.implementation_path)
            adapter = getattr(module, contract.adapter_class)
        except (AttributeError, ImportError, ModuleNotFoundError):
            return "adapter_import_failed"
        if not isinstance(adapter, type) or not issubclass(adapter, ComponentAdapter):
            return "adapter_type_invalid"
        if inspect.getsourcefile(adapter) is None:
            return "adapter_source_unavailable"
        return None

    def _compatibility_errors(self, contract: ComponentContract) -> list[str]:
        errors: list[str] = []
        if contract.fixed_imgsz_compatible is not True:
            errors.append("compatibility_imgsz_640_failed")
        families = set(contract.supported_detector_families)
        if "yolo26" not in families and "generic" not in families:
            errors.append("compatibility_yolo26_family_failed")
        heads = set(contract.supported_heads)
        if "one_to_one" not in heads and "generic" not in heads:
            errors.append("compatibility_yolo26_head_failed")
        if "one_to_one" in contract.incompatible_heads:
            errors.append("compatibility_one_to_one_head_failed")
        if _contains_true_flag(contract.tensor_input_contract, "requires_dfl"):
            errors.append("compatibility_native_dfl_free_failed")
        if _contains_true_flag(contract.tensor_output_contract, "requires_dfl"):
            errors.append("compatibility_native_dfl_free_failed")
        return _unique(errors)

    def _artifact_ok(self, contract: ComponentContract, stage: str) -> bool:
        return self._latest_artifact(contract, stage) is not None

    def _latest_artifact(
        self,
        contract: ComponentContract,
        stage: str,
    ) -> ComponentMaturityArtifact | None:
        for artifact in reversed(contract.maturity_artifacts):
            if artifact.target_maturity != stage or artifact.status != "passed" or artifact.mock:
                continue
            try:
                artifact.verify()
            except (OSError, ValueError):
                continue
            return artifact
        return None

    def _test_refs(
        self,
        component_id: str,
        adapter_class: str,
    ) -> tuple[list[str], list[str], list[str]]:
        cache_key = (component_id, adapter_class)
        cached = self._test_cache.get(cache_key)
        if cached is not None:
            return cached
        unit: list[str] = []
        smoke: list[str] = []
        compatibility: list[str] = []
        if self.tests_root is not None and self.tests_root.is_dir():
            for path in sorted(self.tests_root.rglob("test_*.py")):
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    continue
                lower = text.casefold()
                if component_id.casefold() not in lower and adapter_class.casefold() not in lower:
                    continue
                reference = self._test_reference(path)
                unit.append(reference)
                if any(term in lower for term in _SMOKE_TERMS):
                    smoke.append(reference)
                if any(term in lower for term in _COMPATIBILITY_TERMS):
                    compatibility.append(reference)
        result = (_unique(unit), _unique(smoke), _unique(compatibility))
        self._test_cache[cache_key] = result
        return result

    @staticmethod
    def _test_reference(path: Path) -> str:
        try:
            return path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path.as_posix()

    def _paper_evidence_refs(self, profile: PaperMethodProfile | None) -> list[str]:
        if profile is None:
            return []
        values = list(profile.source_locations)
        if profile.evidence_inventory.note_path:
            values.append(profile.evidence_inventory.note_path)
        return _unique(values)

    @staticmethod
    def _official_code(
        profile: PaperMethodProfile | None,
    ) -> tuple[list[str], str | None]:
        if profile is None:
            return [], None
        metadata = profile.official_code_metadata
        refs = [metadata.url] if metadata.available and metadata.url else []
        license_name = metadata.license if metadata.license != "unknown" else None
        return _unique(refs), license_name

    @staticmethod
    def _required_assets(
        profile: PaperMethodProfile | None,
        inventory: PaperExecutionSpec | None,
    ) -> list[str]:
        values: list[str] = []
        if profile is not None:
            values.extend(profile.protocol_constraints.get("datasets", []))
            values.extend(profile.paper_parameters.get("required_assets", []))
        if inventory is not None:
            values.extend(inventory.required_checkpoints)
            values.extend(inventory.required_evidence)
        return _unique(values)

    @staticmethod
    def _hyperparameters(profile: PaperMethodProfile | None) -> dict[str, Any]:
        if profile is None:
            return {}
        return dict(profile.paper_parameters)

    @staticmethod
    def _implementation_domain(profile: PaperMethodProfile | None) -> str:
        if profile is None:
            return "paper_method"
        family = profile.paper_detector_family or "unspecified_detector"
        return f"{family}:{profile.paper_applicability}"

    @staticmethod
    def _mechanism_summary(
        profile: PaperMethodProfile | None,
        specific_mechanisms: list[str],
    ) -> str:
        if specific_mechanisms:
            return "; ".join(specific_mechanisms)
        if profile is not None and profile.paper_component_ids:
            return "paper-specific mechanism is not explicit; observed terms: " + ", ".join(
                profile.paper_component_ids
            )
        return "paper-specific mechanism is not explicit in the current profile"

    @staticmethod
    def _readiness_level(
        *,
        profile: PaperMethodProfile | None,
        specific_mechanisms: list[str],
        paper_config: dict[str, Any],
        component_ids: list[str],
        adapter_ids: list[str],
        insertion_points: list[str],
        runtime_hooks: list[str],
        runtime_verified: bool,
        unit_passed: bool,
        smoke_passed: bool,
        compatibility_passed: bool,
        evidence_class: ImplementationEvidenceClass,
        blockers: list[str],
    ) -> PaperImplementationReadiness:
        if profile is None:
            return "cataloged"
        if not specific_mechanisms:
            return "profiled"
        spec_complete = bool(
            paper_config
            and component_ids
            and adapter_ids
            and insertion_points
            and runtime_hooks
        )
        if not spec_complete:
            return "profiled"
        if not runtime_verified:
            return "code_bound"
        if not unit_passed:
            return "runtime_integrated"
        if not smoke_passed:
            return "unit_tested"
        if not compatibility_passed:
            return "smoke_passed"
        if evidence_class != "paper_specific" or blockers:
            return "smoke_passed"
        return "implementation_ready"


def _contains_true_flag(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        if value.get(key) is True:
            return True
        return any(_contains_true_flag(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_true_flag(item, key) for item in value)
    return False


def evaluate_paper_implementation(
    *,
    evaluator: PaperImplementationReadinessEvaluator,
    paper: Paper83Paper,
    profile: PaperMethodProfile | None,
    decision: PaperImplementationDecision | None = None,
    inventory: PaperExecutionSpec | None = None,
    coverage: PaperExecutableCoverageEntry | None = None,
) -> PaperImplementationSpec:
    """Functional wrapper used by callers that do not need the evaluator object."""

    return evaluator.evaluate(
        paper=paper,
        profile=profile,
        decision=decision,
        inventory=inventory,
        coverage=coverage,
    )


__all__ = [
    "PaperImplementationReadinessEvaluator",
    "evaluate_paper_implementation",
]
