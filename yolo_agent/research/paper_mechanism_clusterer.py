"""Evidence-grounded clustering of paper methods into reusable runtimes."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from yolo_agent.research.component_aliases import normalize_component_id
from yolo_agent.research.mechanism_clusters import (
    ClusterEvidence,
    MechanismClusterConfig,
    MechanismClusterConflict,
    MechanismClusterDefinition,
    PaperMechanismClusterMatch,
)
from yolo_agent.research.method_profiles import (
    PaperImplementationDecision,
    PaperMethodCoverageReport,
    PaperMethodProfile,
)


@dataclass(frozen=True)
class _Candidate:
    cluster: MechanismClusterDefinition
    evidence: tuple[ClusterEvidence, ...]
    exact_components: tuple[str, ...]
    exact_is_unique: bool
    score: float


class PaperMechanismClusterer:
    """Map profiles to clusters without treating name similarity as runtime identity."""

    def __init__(self, config: MechanismClusterConfig | None = None) -> None:
        self.config = config or MechanismClusterConfig.from_yaml()
        self.by_component: dict[str, list[MechanismClusterDefinition]] = defaultdict(list)
        for cluster in self.config.clusters:
            for component_id in cluster.canonical_component_ids:
                self.by_component[component_id].append(cluster)

    def match_profile(
        self,
        profile: PaperMethodProfile,
        decision: PaperImplementationDecision,
    ) -> tuple[list[PaperMechanismClusterMatch], list[MechanismClusterConflict]]:
        candidates = [
            candidate
            for cluster in self.config.clusters
            if (candidate := self._candidate(profile, cluster)) is not None
        ]
        selected, conflicts = self._select(profile, candidates)
        if not selected:
            return [PaperMechanismClusterMatch(
                paper_id=profile.paper_id,
                profile_id=profile.profile_id,
                match_type="unresolved",
                confidence="low",
                confidence_score=0.0,
                match_reason="no_semantically_compatible_mechanism_cluster",
                conflicts=[item.reason for item in conflicts],
            )], conflicts
        matches = [
            _materialize_match(profile, decision, candidate)
            for candidate in selected
        ]
        return matches, conflicts

    def cluster(
        self,
        coverage: PaperMethodCoverageReport,
    ) -> tuple[list[PaperMechanismClusterMatch], list[MechanismClusterConflict]]:
        decisions = {item.paper_id: item for item in coverage.decisions}
        matches: list[PaperMechanismClusterMatch] = []
        conflicts: list[MechanismClusterConflict] = []
        for profile in sorted(coverage.profiles, key=lambda item: item.paper_id):
            profile_matches, profile_conflicts = self.match_profile(
                profile,
                decisions[profile.paper_id],
            )
            matches.extend(profile_matches)
            conflicts.extend(profile_conflicts)
        return matches, conflicts

    def _candidate(
        self,
        profile: PaperMethodProfile,
        cluster: MechanismClusterDefinition,
    ) -> _Candidate | None:
        evidence = _profile_evidence(profile)
        exact_components = tuple(sorted(
            set(profile.canonical_component_ids)
            & set(cluster.canonical_component_ids)
        ))
        matched: list[ClusterEvidence] = []
        for item in evidence:
            if _evidence_matches_cluster(item, cluster):
                matched.append(item)
        if not exact_components and not matched:
            return None
        exact_score = 0.72 if exact_components else 0.0
        evidence_score = sum(_evidence_weight(item) for item in matched)
        return _Candidate(
            cluster=cluster,
            evidence=tuple(_deduplicate_evidence(matched)),
            exact_components=exact_components,
            exact_is_unique=bool(exact_components) and all(
                len(self.by_component[component_id]) == 1
                for component_id in exact_components
            ),
            score=min(1.0, exact_score + evidence_score),
        )

    def _select(
        self,
        profile: PaperMethodProfile,
        candidates: list[_Candidate],
    ) -> tuple[list[_Candidate], list[MechanismClusterConflict]]:
        if not candidates:
            return [], []
        conflicts: list[MechanismClusterConflict] = []
        by_component: dict[str, list[_Candidate]] = defaultdict(list)
        semantic_only: list[_Candidate] = []
        for candidate in candidates:
            if candidate.exact_components:
                for component in candidate.exact_components:
                    by_component[component].append(candidate)
            else:
                semantic_only.append(candidate)
        selected: dict[str, _Candidate] = {}
        for component_id, component_candidates in sorted(by_component.items()):
            if len(component_candidates) == 1:
                item = component_candidates[0]
                selected[item.cluster.cluster_id] = item
                continue
            resolved = _resolve_ambiguous_component(component_candidates)
            if resolved is None:
                conflicts.append(_conflict(
                    profile,
                    component_candidates,
                    f"ambiguous_training_semantics_for_component:{component_id}",
                ))
                continue
            selected[resolved.cluster.cluster_id] = resolved
        for candidate in semantic_only:
            if candidate.score < 0.55:
                continue
            selected[candidate.cluster.cluster_id] = candidate
        return sorted(
            selected.values(),
            key=lambda item: item.cluster.cluster_id,
        ), conflicts


def _profile_evidence(profile: PaperMethodProfile) -> list[ClusterEvidence]:
    result: list[ClusterEvidence] = []
    structured = profile.structured_method_evidence
    if structured is not None:
        for item in structured.observations:
            result.append(ClusterEvidence(
                field_name=item.field_name,
                value=item.value,
                source=item.source,
                source_location=item.source_location,
                confidence=item.confidence,
            ))
    for item in profile.mechanism_evidence:
        result.append(ClusterEvidence(
            field_name="mechanism_source_term",
            value=item.source_term,
            source=item.source,
            source_location=item.source_location,
            confidence=(
                "low" if item.source == "title"
                else "medium" if item.source in {"harness_hint", "official_code_metadata"}
                else "high"
            ),
        ))
    return _deduplicate_evidence(result)


def _evidence_matches_cluster(
    evidence: ClusterEvidence,
    cluster: MechanismClusterDefinition,
) -> bool:
    if isinstance(evidence.value, bool):
        if evidence.field_name == "training_only":
            return cluster.training_only is evidence.value
        if evidence.field_name == "inference_changed":
            return cluster.inference_changed is evidence.value
        return False
    value = normalize_component_id(evidence.value)
    if evidence.field_name == "canonical_mechanism":
        return evidence.value in cluster.canonical_component_ids
    if evidence.field_name == "method_family":
        return evidence.value in cluster.method_families
    if evidence.field_name == "insertion_point":
        return evidence.value in cluster.insertion_points
    if evidence.field_name == "required_runtime_hook":
        return evidence.value in cluster.required_runtime_hooks
    if evidence.field_name == "changed_variable":
        return evidence.value in cluster.parameter_keys
    if evidence.field_name in {"component_type", "detector_family"}:
        return False
    if evidence.field_name != "mechanism_source_term":
        return False
    terms = [
        *cluster.semantic_aliases,
        *cluster.method_families,
        cluster.display_name,
        cluster.cluster_id,
    ]
    return any(_contains_term(value, normalize_component_id(term)) for term in terms)


def _contains_term(text: str, term: str) -> bool:
    return bool(term and re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", text))


def _evidence_weight(evidence: ClusterEvidence) -> float:
    confidence = {"low": 0.03, "medium": 0.10, "high": 0.18}[evidence.confidence]
    specificity = {
        "mechanism_source_term": 1.5,
        "method_family": 1.25,
        "insertion_point": 0.7,
        "required_runtime_hook": 0.5,
        "training_only": 0.3,
        "inference_changed": 0.3,
    }.get(evidence.field_name, 1.0)
    return confidence * specificity


def _resolve_ambiguous_component(candidates: list[_Candidate]) -> _Candidate | None:
    discriminating_fields = {
        "mechanism_source_term",
        "insertion_point",
        "changed_variable",
    }
    specificity = {
        item.cluster.cluster_id: sum(
            _evidence_weight(evidence)
            for evidence in item.evidence
            if evidence.field_name in discriminating_fields
            and evidence.confidence in {"medium", "high"}
        )
        for item in candidates
    }
    ranked = sorted(
        candidates,
        key=lambda item: (
            -specificity[item.cluster.cluster_id],
            -item.score,
            item.cluster.cluster_id,
        ),
    )
    best = ranked[0]
    second = ranked[1]
    if (
        specificity[best.cluster.cluster_id] == 0
        or specificity[best.cluster.cluster_id]
        - specificity[second.cluster.cluster_id] < 0.10
    ):
        return None
    return best


def _materialize_match(
    profile: PaperMethodProfile,
    decision: PaperImplementationDecision,
    candidate: _Candidate,
) -> PaperMechanismClusterMatch:
    exact = candidate.exact_is_unique
    match_type = "exact_match" if exact else "semantic_match"
    score = candidate.score
    confidence = "high" if score >= 0.85 else "medium" if score >= 0.55 else "low"
    adapter_available = any(
        item.canonical_component_id in candidate.cluster.canonical_component_ids
        and item.adapter_verified
        for item in decision.mechanism_mappings
    )
    runtime_ready = any(
        item.canonical_component_id in candidate.cluster.canonical_component_ids
        and item.runtime_execution_ready
        for item in decision.mechanism_mappings
    )
    reason = (
        "unique canonical component maps to one runtime semantic"
        if match_type == "exact_match"
        else "source-grounded semantic evidence disambiguates runtime behavior"
    )
    return PaperMechanismClusterMatch(
        paper_id=profile.paper_id,
        profile_id=profile.profile_id,
        cluster_id=candidate.cluster.cluster_id,
        adapter_family=candidate.cluster.adapter_family,
        training_semantic=candidate.cluster.training_semantic,
        match_type=match_type,
        confidence=confidence,
        confidence_score=score,
        evidence=list(candidate.evidence),
        match_reason=reason,
        adapter_available=adapter_available,
        runtime_ready=runtime_ready,
    )


def _conflict(
    profile: PaperMethodProfile,
    candidates: list[_Candidate],
    reason: str,
) -> MechanismClusterConflict:
    return MechanismClusterConflict(
        paper_id=profile.paper_id,
        candidate_cluster_ids=sorted({item.cluster.cluster_id for item in candidates}),
        reason=reason,
        evidence_locations=sorted({
            evidence.source_location
            for item in candidates
            for evidence in item.evidence
        }),
    )


def _deduplicate_evidence(evidence: list[ClusterEvidence]) -> list[ClusterEvidence]:
    unique = {
        (
            item.field_name,
            str(item.value),
            item.source,
            item.source_location,
            item.confidence,
        ): item
        for item in evidence
    }
    return sorted(
        unique.values(),
        key=lambda item: (
            item.field_name,
            str(item.value),
            item.source,
            item.source_location,
        ),
    )


__all__ = ["PaperMechanismClusterer"]
