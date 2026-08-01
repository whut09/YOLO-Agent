"""Build executable paper coverage without granting implementation maturity."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Mapping

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.executable_coverage_schemas import (
    AdaptationScope,
    CompatibilityClass,
    ExecutablePaperCoverageBaseline,
    PaperCoverageDenominator,
    PaperExecutableCoverageEntry,
    PaperExpectedResourceCost,
    PaperImplementationCost,
)
from yolo_agent.research.maturity_snapshot import EffectiveComponentMaturityManifest
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)


_DENOMINATOR_DEFINITIONS = {
    "all_papers": "Every unique PaperRecord represented by one MethodProfile and implementation decision.",
    "yolo26_compatible_papers": (
        "Papers with at least one explicitly mapped YOLO26-compatible or "
        "adapter-required mechanism, excluding separate detector families."
    ),
    "adaptable_component_papers": (
        "YOLO26-compatible papers whose scoped method can be adapted as one or "
        "more isolated components rather than replacing the detector family."
    ),
    "exact_reproduction_candidates": (
        "Papers explicitly marked for exact reproduction with sufficient protocol "
        "and runtime evidence; component adaptation alone never qualifies."
    ),
}


class ExecutablePaperCoverageAuditor:
    """Audit one frozen method report against explicit runtime identities."""

    def __init__(
        self,
        *,
        contracts: Mapping[str, ComponentContract] | None = None,
        maturity: EffectiveComponentMaturityManifest | None = None,
    ) -> None:
        self.contracts = dict(contracts or {})
        self.maturity = maturity or EffectiveComponentMaturityManifest()
        self.runtime_identities = self.maturity.by_component()

    def build(
        self,
        method_coverage: PaperMethodCoverageReport,
        *,
        source_method_coverage_hash: str,
        source_taxonomy_hash: str,
    ) -> ExecutablePaperCoverageBaseline:
        profiles = {item.paper_id: item for item in method_coverage.profiles}
        decisions = {item.paper_id: item for item in method_coverage.decisions}
        paper_ids = sorted(set(profiles) | set(decisions))
        if len(profiles) != method_coverage.profile_count:
            raise ValueError("method coverage contains duplicate paper profiles")
        if len(decisions) != method_coverage.paper_count:
            raise ValueError("method coverage must contain one decision per paper")
        entries = [
            self._entry(profiles[paper_id], decisions[paper_id])
            for paper_id in paper_ids
        ]
        denominator_members = {
            "all_papers": [item.paper_id for item in entries],
            "yolo26_compatible_papers": [
                item.paper_id for item in entries if _is_yolo26_compatible(item)
            ],
            "adaptable_component_papers": [
                item.paper_id for item in entries if _is_adaptable_component(item)
            ],
            "exact_reproduction_candidates": [
                item.paper_id for item in entries if item.exact_reproduction_possible
            ],
        }
        denominators = {
            name: PaperCoverageDenominator(
                name=name,  # type: ignore[arg-type]
                definition=_DENOMINATOR_DEFINITIONS[name],
                paper_count=len(ids),
                paper_ids=ids,
            )
            for name, ids in denominator_members.items()
        }
        return ExecutablePaperCoverageBaseline(
            source_method_coverage_hash=source_method_coverage_hash,
            source_taxonomy_hash=source_taxonomy_hash,
            source_maturity_hash=(
                self.maturity.manifest_hash if self.maturity.entries else None
            ),
            denominators=denominators,
            compatibility_counts=dict(
                sorted(Counter(item.compatibility_class for item in entries).items())
            ),
            runtime_ready_paper_count=sum(
                bool(item.runtime_ready_adapters) for item in entries
            ),
            reusable_adapter_paper_count=sum(
                bool(item.reusable_adapter_candidates) for item in entries
            ),
            mechanism_to_papers=_reverse_index(
                entries, "canonical_mechanisms"
            ),
            adapter_to_papers=_reverse_index(
                entries, "reusable_adapter_candidates"
            ),
            runtime_adapter_to_papers=_reverse_index(
                entries, "runtime_ready_adapters"
            ),
            entries=entries,
        )

    def _entry(
        self,
        profile: PaperMethodProfile,
        decision: PaperImplementationDecision,
    ) -> PaperExecutableCoverageEntry:
        mechanisms = sorted(set(decision.canonical_component_ids))
        reusable = sorted(set(decision.reusable_adapter_ids))
        runtime_ready = sorted(
            component_id
            for component_id in reusable
            if self._runtime_ready(component_id)
        )
        blockers = _blocking_fields(profile, decision, mechanisms, reusable)
        compatibility = _compatibility_class(
            profile,
            decision,
            mechanisms=mechanisms,
            reusable=reusable,
            runtime_ready=runtime_ready,
        )
        scope = _adaptation_scope(profile, decision, mechanisms)
        hooks = sorted(
            {
                hook
                for component_id in mechanisms
                for hook in _runtime_hooks(self.contracts.get(component_id))
            }
        )
        exact_possible = bool(
            profile.exact_reproduction_claim
            and decision.exact_reproduction_claim
            and decision.adaptation_mode == "exact_reproduction"
            and not blockers
            and reusable
            and set(reusable).issubset(runtime_ready)
        )
        exclusion = _exclusion_reason(
            compatibility,
            blockers=blockers,
            exact_reproduction_possible=exact_possible,
            profile=profile,
        )
        return PaperExecutableCoverageEntry(
            paper_id=profile.paper_id,
            profile_id=profile.profile_id,
            decision=decision.decision,
            compatibility_class=compatibility,
            adaptation_scope=scope,
            blocking_fields=blockers,
            canonical_mechanisms=mechanisms,
            reusable_adapter_candidates=reusable,
            runtime_ready_adapters=runtime_ready,
            required_runtime_hooks=hooks,
            implementation_cost=_implementation_cost(
                mechanisms,
                reusable,
                hooks,
            ),
            expected_resource_cost=_resource_cost(
                mechanisms,
                self.contracts,
            ),
            exact_reproduction_possible=exact_possible,
            exclusion_reason=exclusion,
            source_locations=sorted(
                set(profile.source_locations) | set(decision.source_locations)
            ),
        )

    def _runtime_ready(self, component_id: str) -> bool:
        identity = self.runtime_identities.get(component_id)
        return bool(identity and identity.runtime_execution_ready)


def method_coverage_file_hash(path: Path | str) -> str:
    """Return the byte identity used to bind the derived baseline."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compatibility_class(
    profile: PaperMethodProfile,
    decision: PaperImplementationDecision,
    *,
    mechanisms: list[str],
    reusable: list[str],
    runtime_ready: list[str],
) -> CompatibilityClass:
    if decision.decision == "separate_detector_family":
        return "separate_detector_family"
    if decision.decision == "insufficient_information" or not mechanisms:
        return "insufficient_information"
    mapping_compatibility = {
        item.yolo26_compatibility for item in decision.mechanism_mappings
    }
    if mapping_compatibility and mapping_compatibility <= {"incompatible"}:
        return "incompatible"
    if decision.decision == "coupled_recipe":
        return "yolo26_coupled_adaptation"
    if runtime_ready:
        return "yolo26_runtime_ready"
    if reusable:
        return "yolo26_adapter_available"
    if profile.adaptation_mode == "component_adaptation":
        return "yolo26_adapter_required"
    return "insufficient_information"


def _adaptation_scope(
    profile: PaperMethodProfile,
    decision: PaperImplementationDecision,
    mechanisms: list[str],
) -> AdaptationScope:
    if profile.adaptation_mode == "exact_reproduction":
        return "exact_reproduction"
    if decision.decision == "separate_detector_family":
        return "whole_detector"
    if decision.decision == "coupled_recipe":
        return "coupled_components"
    if len(mechanisms) > 1:
        return "multiple_independent_components"
    if len(mechanisms) == 1:
        return "single_component"
    return "none"


def _blocking_fields(
    profile: PaperMethodProfile,
    decision: PaperImplementationDecision,
    mechanisms: list[str],
    reusable: list[str],
) -> list[str]:
    blockers = {
        f"{item.field_name}:{item.reason_code}"
        for item in decision.adaptation_gaps
        if item.severity == "blocking"
    }
    if not mechanisms:
        blockers.add("canonical_mechanisms:missing")
    if mechanisms and not reusable:
        blockers.add("reusable_adapter:missing")
    if profile.adaptation_mode == "exact_reproduction":
        if not profile.protocol_constraints:
            blockers.add("protocol_constraints:missing")
        if not profile.official_code_metadata.available:
            blockers.add("official_code:missing")
    return sorted(blockers)


def _runtime_hooks(contract: ComponentContract | None) -> list[str]:
    if contract is None:
        return []
    insertion = contract.insertion_point.lower()
    category = contract.category.lower()
    hooks: set[str] = set()
    if any(term in insertion or term in category for term in ("data", "sampl")):
        hooks.add("build_train_dataloader")
    if any(term in insertion or term in category for term in ("loss", "criterion")):
        hooks.update({"build_criterion", "compute_loss"})
    if any(term in insertion or term in category for term in ("head", "neck", "graph")):
        hooks.add("build_model")
    if "assign" in insertion or "assign" in category:
        hooks.add("build_criterion")
    if contract.inference_only is True or "inference" in category:
        hooks.add("build_validator")
    return sorted(hooks)


def _implementation_cost(
    mechanisms: list[str],
    reusable: list[str],
    hooks: list[str],
) -> PaperImplementationCost:
    missing = max(len(mechanisms) - len(reusable), 0)
    if not mechanisms:
        level = "unknown"
    elif missing == 0 and len(hooks) <= 1:
        level = "low"
    elif missing <= 1 and len(hooks) <= 2:
        level = "medium"
    else:
        level = "high"
    return PaperImplementationCost(
        level=level,
        adapter_count=len(reusable),
        missing_adapter_count=missing,
        hook_count=len(hooks),
        rationale=[
            f"canonical_mechanisms={len(mechanisms)}",
            f"reusable_adapters={len(reusable)}",
            f"required_runtime_hooks={len(hooks)}",
        ],
    )


def _resource_cost(
    mechanisms: list[str],
    contracts: Mapping[str, ComponentContract],
) -> PaperExpectedResourceCost:
    selected = [contracts[item] for item in mechanisms if item in contracts]
    if not selected:
        return PaperExpectedResourceCost(
            rationale=["no local component contract declares resource impact"]
        )
    latency = _combine_declared(item.affects_latency for item in selected)
    model_size = _combine_declared(item.affects_model_size for item in selected)
    graph_change = any(item.changes_model_graph is True for item in selected)
    training_only = all(item.training_only is True for item in selected)
    level: str = "high" if graph_change else "low" if training_only else "medium"
    return PaperExpectedResourceCost(
        level=level,  # type: ignore[arg-type]
        latency=latency,
        model_size=model_size,
        vram="increased" if graph_change else "unknown",
        training_compute="increased" if selected else "unknown",
        rationale=[
            "resource values come from local component contracts",
            "no paper benchmark is treated as local resource evidence",
        ],
    )


def _combine_declared(values: object) -> str:
    declared = {str(item) for item in values if str(item) != "unknown"}
    return ",".join(sorted(declared)) if declared else "unknown"


def _exclusion_reason(
    compatibility: CompatibilityClass,
    *,
    blockers: list[str],
    exact_reproduction_possible: bool,
    profile: PaperMethodProfile,
) -> str | None:
    if compatibility == "separate_detector_family":
        return "paper requires a separate detector-family track"
    if compatibility == "incompatible":
        return "mapped mechanisms are incompatible with YOLO26"
    if compatibility == "insufficient_information":
        return "local paper evidence is insufficient for component adaptation"
    if blockers:
        return "; ".join(blockers)
    if profile.exact_reproduction_claim and not exact_reproduction_possible:
        return "exact reproduction lacks complete protocol or runtime evidence"
    return None


def _is_yolo26_compatible(item: PaperExecutableCoverageEntry) -> bool:
    return item.compatibility_class in {
        "yolo26_runtime_ready",
        "yolo26_adapter_available",
        "yolo26_adapter_required",
        "yolo26_coupled_adaptation",
    }


def _is_adaptable_component(item: PaperExecutableCoverageEntry) -> bool:
    return _is_yolo26_compatible(item) and item.adaptation_scope in {
        "single_component",
        "multiple_independent_components",
        "coupled_components",
    }


def _reverse_index(
    entries: list[PaperExecutableCoverageEntry],
    field: str,
) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = {}
    for entry in entries:
        for value in getattr(entry, field):
            grouped.setdefault(value, set()).add(entry.paper_id)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


__all__ = ["ExecutablePaperCoverageAuditor", "method_coverage_file_hash"]
