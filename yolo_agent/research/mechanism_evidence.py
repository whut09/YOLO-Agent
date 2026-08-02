"""Evidence-grounded canonical mechanism mentions from local paper text."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from yolo_agent.research.code_metadata import parse_official_code_metadata
from yolo_agent.research.component_aliases import (
    ComponentAliasResolver,
    normalize_component_id,
)
from yolo_agent.research.schemas import PaperRecord


MechanismEvidenceSource = Literal[
    "catalog_component_id",
    "title",
    "summary",
    "note",
    "harness_hint",
    "official_code_metadata",
]


class PaperMechanismEvidence(BaseModel):
    """One explicit text-to-canonical-mechanism observation."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source_term: str
    canonical_component_id: str
    source: MechanismEvidenceSource
    source_location: str
    alias_match_type: str
    evidence_level: Literal["paper_prior"] = "paper_prior"


class MechanismEvidenceExtractor:
    """Match only curated mechanism aliases explicitly present in local text."""

    def __init__(self, resolver: ComponentAliasResolver) -> None:
        self.resolver = resolver
        self.terms = _curated_terms(resolver)

    def extract(
        self,
        paper: PaperRecord,
        *,
        evidence_summary: Any | None = None,
    ) -> list[PaperMechanismEvidence]:
        observations: list[PaperMechanismEvidence] = []
        for component_id in paper.component_ids:
            observations.extend(self._resolve_term(
                paper.paper_id,
                component_id,
                source="catalog_component_id",
                source_location="paper_record.component_ids",
            ))
        sources: list[tuple[MechanismEvidenceSource, str, str]] = []
        if paper.title:
            sources.append(("title", "paper_record.title", paper.title))
        if paper.abstract:
            sources.append(("summary", "summary", paper.abstract))
        if evidence_summary is not None:
            for claim in getattr(evidence_summary, "method_claims", []) or []:
                sources.append((
                    "note" if str(claim.source_location).startswith("note") else "summary",
                    str(claim.source_location),
                    " ".join([claim.method_name, *claim.component_ids]),
                ))
            for hint in getattr(evidence_summary, "diagnostic_hints", []) or []:
                sources.append((
                    "harness_hint",
                    str(hint.source_location),
                    " ".join([
                        hint.symptom,
                        hint.likely_cause,
                        *hint.candidate_component_ids,
                    ]),
                ))
        provenance = paper.provenance
        if provenance is not None:
            for index, hint in enumerate(provenance.original_harness_hints):
                sources.append(("harness_hint", f"harness_hints[{index}]", hint))
        code = parse_official_code_metadata(paper)
        if code.repository_slug:
            sources.append((
                "official_code_metadata",
                "paper_record.official_code_url",
                code.repository_slug,
            ))
        for source, location, text in sources:
            normalized_text = normalize_component_id(text)
            for term in _most_specific_terms(normalized_text, self.terms):
                observations.extend(self._resolve_term(
                    paper.paper_id,
                    term,
                    source=source,
                    source_location=location,
                ))
        unique = {
            (
                item.canonical_component_id,
                item.source,
                item.source_location,
                item.source_term,
            ): item
            for item in observations
        }
        return sorted(
            unique.values(),
            key=lambda item: (
                item.canonical_component_id,
                item.source,
                item.source_location,
                item.source_term,
            ),
        )

    def _resolve_term(
        self,
        paper_id: str,
        term: str,
        *,
        source: MechanismEvidenceSource,
        source_location: str,
    ) -> list[PaperMechanismEvidence]:
        resolution = self.resolver.resolve(term, source_paper_ids=[paper_id])
        return [
            PaperMechanismEvidence(
                paper_id=paper_id,
                source_term=term,
                canonical_component_id=mapping.canonical_component_id,
                source=source,
                source_location=source_location,
                alias_match_type=resolution.match_type,
            )
            for mapping in resolution.mappings
        ]


def _curated_terms(resolver: ComponentAliasResolver) -> list[str]:
    terms: set[str] = set()
    for definition in resolver.config.canonical_components:
        terms.update(definition.aliases)
        terms.update(definition.semantic_aliases)
    for definition in resolver.config.compound_aliases:
        terms.update(definition.aliases)
        terms.update(definition.semantic_aliases)
    return sorted(
        {normalize_component_id(term) for term in terms if term.strip()},
        key=lambda item: (-len(item), item),
    )


def _contains_normalized_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?:^|_){re.escape(term)}(?:_|$)", text))


def _most_specific_terms(text: str, terms: list[str]) -> list[str]:
    selected: list[str] = []
    for term in terms:
        if not _contains_normalized_term(text, term):
            continue
        if any(_contains_normalized_term(existing, term) for existing in selected):
            continue
        selected.append(term)
    return selected


__all__ = [
    "MechanismEvidenceExtractor",
    "MechanismEvidenceSource",
    "PaperMechanismEvidence",
]
