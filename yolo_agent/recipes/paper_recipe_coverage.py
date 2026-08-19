"""Coverage report for certified paper-to-recipe bindings."""

from __future__ import annotations

from collections import Counter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.yaml_io import YAMLModelMixin
from yolo_agent.recipes.paper_recipe_bindings import (
    bindings_by_paper_id,
    load_certified_paper_recipe_specs,
)
from yolo_agent.recipes.paper_recipe_spec import PaperRecipeDisposition, PaperRecipeSpec
from yolo_agent.research.paper_mechanism_resolver import GENERIC_MECHANISM_IDS
from yolo_agent.research.paper_protocol_ids import CERTIFIED_PAPER_MECHANISMS


UNRESOLVED_DISPOSITIONS = {
    "evidence_recovery",
    "implementation_request",
    "incompatible",
    "blocked_runtime",
}


class PaperRecipeBindingRecord(BaseModel):
    """One auditable paper-to-recipe row."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    recipe_id: str
    paper_specific_mechanism_id: str
    disposition: PaperRecipeDisposition
    execution_fingerprint: str
    protocol_hash: str
    reason_codes: list[str] = Field(default_factory=list)


class PaperRecipeCoverageReport(BaseModel, YAMLModelMixin):
    """Machine-readable coverage of the 83 certified papers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_recipe_coverage.v1"
    papers_total: int
    paper_recipe_bindings_total: int
    queued: int = 0
    evidence_recovery: int = 0
    implementation_request: int = 0
    incompatible: int = 0
    blocked_runtime: int = 0
    unresolved_bindings: list[PaperRecipeBindingRecord] = Field(default_factory=list)
    bindings: list[PaperRecipeBindingRecord] = Field(default_factory=list)
    generic_collapse: list[str] = Field(default_factory=list)
    silent_drops: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_coverage(self) -> "PaperRecipeCoverageReport":
        if self.silent_drops:
            raise ValueError(f"paper recipe coverage has silent drops: {self.silent_drops}")
        if self.generic_collapse:
            raise ValueError(f"generic recipe collapse is forbidden: {self.generic_collapse}")
        if self.papers_total != self.paper_recipe_bindings_total:
            raise ValueError("every paper must have exactly one recipe binding")
        for row in self.unresolved_bindings:
            if row.disposition not in UNRESOLVED_DISPOSITIONS:
                raise ValueError(f"unresolved binding {row.paper_id} lacks a terminal disposition")
        return self


def build_paper_recipe_coverage(
    specs: list[PaperRecipeSpec] | None = None,
) -> PaperRecipeCoverageReport:
    """Build the 83-paper recipe coverage report with no silent drops."""
    loaded = list(specs or load_certified_paper_recipe_specs())
    by_paper = bindings_by_paper_id(loaded)
    certified = list(CERTIFIED_PAPER_MECHANISMS)
    silent = [paper_id for paper_id in certified if paper_id not in by_paper]
    rows = [
        PaperRecipeBindingRecord(
            paper_id=paper_id,
            recipe_id=by_paper[paper_id].recipe_id,
            paper_specific_mechanism_id=by_paper[paper_id].paper_specific_mechanism_id,
            disposition=by_paper[paper_id].disposition,
            execution_fingerprint=by_paper[paper_id].execution_fingerprint,
            protocol_hash=by_paper[paper_id].protocol_hash,
            reason_codes=_reason_codes(by_paper[paper_id]),
        )
        for paper_id in certified
        if paper_id in by_paper
    ]
    counts = Counter(row.disposition for row in rows)
    unresolved = [row for row in rows if row.disposition != "queued"]
    return PaperRecipeCoverageReport(
        papers_total=len(certified),
        paper_recipe_bindings_total=len(rows),
        queued=counts.get("queued", 0),
        evidence_recovery=counts.get("evidence_recovery", 0),
        implementation_request=counts.get("implementation_request", 0),
        incompatible=counts.get("incompatible", 0),
        blocked_runtime=counts.get("blocked_runtime", 0),
        unresolved_bindings=unresolved,
        bindings=rows,
        generic_collapse=_generic_collapse(loaded),
        silent_drops=silent,
    )


def reject_generic_recipe_collapse(specs: list[PaperRecipeSpec]) -> list[str]:
    """Return collapse violations: one generic recipe covering many papers."""
    return _generic_collapse(specs)


def _generic_collapse(specs: list[PaperRecipeSpec]) -> list[str]:
    by_recipe: dict[str, set[str]] = {}
    by_mechanism: dict[str, set[str]] = {}
    for spec in specs:
        by_recipe.setdefault(spec.recipe_id, set()).update(spec.paper_ids)
        by_mechanism.setdefault(spec.paper_specific_mechanism_id, set()).update(spec.paper_ids)
        if spec.paper_specific_mechanism_id in GENERIC_MECHANISM_IDS:
            return [f"{spec.recipe_id}:{spec.paper_specific_mechanism_id}"]
    violations: list[str] = []
    if len(by_recipe.get("yolo26n_distillation", ())) > 1:
        violations.append("yolo26n_distillation_covers_multiple_papers")
    generic_da_papers = {
        paper_id
        for spec in specs
        if spec.paper_specific_mechanism_id == "domain_adaptation.general"
        for paper_id in spec.paper_ids
    }
    if len(generic_da_papers) > 1:
        violations.append("domain_adaptation.general_covers_multiple_papers")
    generic_kd_papers = {
        paper_id
        for spec in specs
        if spec.paper_specific_mechanism_id == "distillation.yolo26_teacher_student"
        for paper_id in spec.paper_ids
    }
    if len(generic_kd_papers) > 1:
        violations.append("distillation.yolo26_teacher_student_covers_multiple_papers")
    return violations


def _reason_codes(spec: PaperRecipeSpec) -> list[str]:
    if spec.disposition == "queued":
        return []
    if spec.disposition == "evidence_recovery":
        return ["target_error_facts_missing"]
    if spec.inference_only:
        return ["inference_only_not_training_candidate"]
    if spec.disposition == "implementation_request":
        return [f"adapter_required:{spec.runtime_plugin}"]
    return [spec.disposition]
