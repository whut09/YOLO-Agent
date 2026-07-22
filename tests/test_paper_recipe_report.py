from pathlib import Path
from yolo_agent.reports.paper_recipe_report import PaperRecipeReportBuilder, PaperRecipeReportEntry, PaperRecipeReportInput, terminal_summary
from yolo_agent.reports.pareto_report import PaperParetoCandidate


def test_doctor_report_separates_claims_local_evidence_and_contributions(tmp_path: Path) -> None:
    source = PaperRecipeReportInput(
        run_id="run-1",
        current_diagnosis="AP_small and recall are low",
        research_snapshot={"snapshot_hash": "snapshot-1"},
        recipes=[
            PaperRecipeReportEntry(recipe_id="possible", paper_ids=["p1"], component_ids=["sampling.small"], changed_variable="data.sampler", maturity="smoke_passed", compatibility="compatible", implementation_status="smoke_passed", paper_claim={"ap_small": "+1.0"}, local_evidence={"map50_95_delta": 0.004}, pilot_result={"stage": "pilot_10"}, error_delta={"ap_small": 0.01}, evidence_status="possible"),
            PaperRecipeReportEntry(recipe_id="confirmed", paper_ids=["p2"], component_ids=["distillation"], changed_variable="distillation.loss", maturity="pilot_reproduced", compatibility="compatible", implementation_status="pilot_reproduced", paper_claim={"map": "+2.0"}, local_evidence={"map50_95_delta": 0.021, "seed_count": 3}, evidence_status="confirmed"),
            PaperRecipeReportEntry(recipe_id="metadata", paper_ids=["p3"], component_ids=["deformable"], paper_claim={"map": "+3.0"}, implementation_status="metadata_only"),
        ],
        asha_trials=[{"candidate_id": "bad", "eliminated_reason": "pilot_3_non_positive_paired_delta"}],
        pareto_candidates=[PaperParetoCandidate(candidate_id="confirmed-c", recipe_id="confirmed", map50_95=0.42, ap_small=0.26, recall=0.61, latency_ms=5.2, model_size_mb=6.1, verified_local=True, evidence_status="confirmed")],
        full_candidate_recommendations=["confirmed-c"],
        next_step_recommendations=["Confirm full run after consent."],
    )
    report = PaperRecipeReportBuilder().build(source)
    assert report.possible_contributions == ["possible"]
    assert report.confirmed_contributions == ["confirmed"]
    assert report.asha_eliminations["bad"].startswith("pilot_3")
    assert "paper_claim_not_local_evidence" in report.unverified_risks
    yaml_path, md_path = PaperRecipeReportBuilder().write(source, tmp_path)
    assert yaml_path.is_file() and md_path.is_file()
    text = md_path.read_text(encoding="utf-8")
    assert "Paper claim" in text and "Local evidence" in text and "Pareto Front" in text
    summary = terminal_summary(report)
    assert len(summary) == 6
    assert summary[0].startswith("Diagnosis:")


def test_unexecuted_component_is_never_reported_as_reproduced() -> None:
    source = PaperRecipeReportInput(run_id="run", recipes=[PaperRecipeReportEntry(recipe_id="idea", implementation_status="metadata_only", local_evidence={"map50_95_delta": 0.1}, evidence_status="confirmed")])
    report = PaperRecipeReportBuilder().build(source)
    assert "component_not_executed" in report.unverified_risks
