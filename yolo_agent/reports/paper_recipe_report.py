"""Doctor-style paper recipe report with strict evidence provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from yolo_agent.agents.loop_io import write_yaml
from yolo_agent.reports.pareto_report import (
    PaperParetoCandidate,
    PaperParetoReport,
    build_paper_pareto_report,
)


class PaperRecipeReportEntry(BaseModel):
    recipe_id: str
    paper_ids: list[str] = Field(default_factory=list)
    component_ids: list[str] = Field(default_factory=list)
    changed_variable: str = "unknown"
    maturity: str = "metadata_only"
    compatibility: str = "unknown"
    implementation_status: str = "metadata_only"
    paper_claim: dict[str, Any] = Field(default_factory=dict)
    local_evidence: dict[str, Any] = Field(default_factory=dict)
    pilot_result: dict[str, Any] = Field(default_factory=dict)
    error_delta: dict[str, Any] = Field(default_factory=dict)
    latency_delta: float | None = None
    model_size_delta: float | None = None
    rejection_reason: str | None = None
    evidence_status: str = "possible"
    unverified_risks: list[str] = Field(default_factory=list)


class PaperRecipeReportInput(BaseModel):
    run_id: str
    current_diagnosis: str = "unknown"
    research_snapshot: dict[str, Any] = Field(default_factory=dict)
    recipes: list[PaperRecipeReportEntry] = Field(default_factory=list)
    asha_trials: list[dict[str, Any]] = Field(default_factory=list)
    pareto_candidates: list[PaperParetoCandidate] = Field(default_factory=list)
    full_candidate_recommendations: list[str] = Field(default_factory=list)
    next_step_recommendations: list[str] = Field(default_factory=list)


class PaperRecipeReport(BaseModel):
    schema_version: str = "paper_recipe_doctor_report.v1"
    run_id: str
    current_diagnosis: str
    research_snapshot: dict[str, Any]
    recipes: list[PaperRecipeReportEntry]
    asha_eliminations: dict[str, str] = Field(default_factory=dict)
    pareto: PaperParetoReport
    possible_contributions: list[str] = Field(default_factory=list)
    confirmed_contributions: list[str] = Field(default_factory=list)
    full_candidate_recommendations: list[str] = Field(default_factory=list)
    unverified_risks: list[str] = Field(default_factory=list)
    next_step_recommendations: list[str] = Field(default_factory=list)


class PaperRecipeReportBuilder:
    def build(self, source: PaperRecipeReportInput) -> PaperRecipeReport:
        eliminations: dict[str, str] = {}
        for item in source.asha_trials:
            if item.get("eliminated_reason"):
                key = str(item.get("candidate_id") or item.get("trial_id") or "unknown")
                eliminations[key] = str(item["eliminated_reason"])
        possible = [item.recipe_id for item in source.recipes if item.evidence_status == "possible" and item.local_evidence]
        confirmed = [item.recipe_id for item in source.recipes if item.evidence_status == "confirmed" and item.local_evidence]
        risks = {risk for item in source.recipes for risk in item.unverified_risks}
        risks.update("paper_claim_not_local_evidence" for item in source.recipes if item.paper_claim and not item.local_evidence)
        risks.update(
            "component_not_executed"
            for item in source.recipes
            if item.implementation_status in {"metadata_only", "adapter_required", "adapter_implemented"}
            and item.local_evidence
        )
        return PaperRecipeReport(
            run_id=source.run_id,
            current_diagnosis=source.current_diagnosis,
            research_snapshot=source.research_snapshot,
            recipes=source.recipes,
            asha_eliminations=eliminations,
            pareto=build_paper_pareto_report(source.pareto_candidates),
            possible_contributions=possible,
            confirmed_contributions=confirmed,
            full_candidate_recommendations=source.full_candidate_recommendations,
            unverified_risks=sorted(risks),
            next_step_recommendations=source.next_step_recommendations,
        )

    def write(self, source: PaperRecipeReportInput, artifact_dir: Path | str) -> tuple[Path, Path]:
        report = self.build(source)
        root = Path(artifact_dir)
        yaml_path = root / "paper_recipe_report.yaml"
        markdown_path = root / "paper_recipe_report.md"
        write_yaml(yaml_path, report.model_dump(mode="json"))
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        return yaml_path, markdown_path


def terminal_summary(report: PaperRecipeReport) -> list[str]:
    current = report.recipes[-1] if report.recipes else None
    delta = current.local_evidence.get("map50_95_delta") if current else None
    conclusion = "full candidate ready" if report.full_candidate_recommendations else "continue guarded pilot/evidence work"
    return [
        f"Diagnosis: {report.current_diagnosis}",
        f"Recipe: {current.recipe_id if current else 'none'}",
        f"Stage: {current.pilot_result.get('stage', 'none') if current else 'none'}",
        f"Delta: {delta if delta is not None else 'unknown'}",
        f"Elimination: {current.rejection_reason if current and current.rejection_reason else 'none'}",
        f"Conclusion: {conclusion}",
    ]


def _markdown(report: PaperRecipeReport) -> str:
    lines = [
        "# Paper Recipe Report", "", "## Current Diagnosis", report.current_diagnosis,
        "", "## Research Snapshot", f"- {report.research_snapshot.get('snapshot_hash', 'unavailable')}",
        "", "## Recipes",
    ]
    for item in report.recipes:
        lines.extend([
            f"### {item.recipe_id}",
            f"- Papers: {', '.join(item.paper_ids) or 'none'}",
            f"- Components: {', '.join(item.component_ids) or 'none'}",
            f"- Changed variable: {item.changed_variable}",
            f"- Maturity / compatibility: {item.maturity} / {item.compatibility}",
            f"- Paper claim: {item.paper_claim or 'none'}",
            f"- Local evidence: {item.local_evidence or 'none'}",
            f"- Pilot result: {item.pilot_result or 'none'}",
            f"- Error delta: {item.error_delta or 'none'}",
            f"- Latency / size delta: {item.latency_delta} / {item.model_size_delta}",
            f"- Status: {item.evidence_status}",
            f"- Rejection: {item.rejection_reason or 'none'}",
        ])
    lines.extend(["", "## ASHA Eliminations"])
    lines.extend([f"- {key}: {value}" for key, value in report.asha_eliminations.items()] or ["- None"])
    lines.extend(["", "## Pareto Front"])
    lines.extend([f"- {item.candidate_id}: {item.tradeoff}" for item in report.pareto.included] or ["- No eligible local training evidence."])
    lines.extend(["", "## Contributions", f"- Possible: {', '.join(report.possible_contributions) or 'none'}", f"- Confirmed: {', '.join(report.confirmed_contributions) or 'none'}"])
    lines.extend(["", "## Full Candidates"])
    lines.extend([f"- {item}" for item in report.full_candidate_recommendations] or ["- None"])
    lines.extend(["", "## Unverified Risks"])
    lines.extend([f"- {item}" for item in report.unverified_risks] or ["- None"])
    lines.extend(["", "## Next Step"])
    lines.extend([f"- {item}" for item in report.next_step_recommendations] or ["- Collect missing local evidence before training."])
    return "\n".join(lines) + "\n"


__all__ = [
    "PaperRecipeReport", "PaperRecipeReportBuilder", "PaperRecipeReportEntry",
    "PaperRecipeReportInput", "terminal_summary",
]
