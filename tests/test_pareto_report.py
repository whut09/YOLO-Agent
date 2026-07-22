from yolo_agent.reports.pareto_report import PaperParetoCandidate, build_paper_pareto_report


def test_pareto_uses_only_current_verified_local_training_evidence() -> None:
    report = build_paper_pareto_report([
        PaperParetoCandidate(candidate_id="good", recipe_id="r1", map50_95=0.42, ap_small=0.25, recall=0.60, latency_ms=5.0, model_size_mb=6.0, verified_local=True),
        PaperParetoCandidate(candidate_id="dominated", recipe_id="r2", map50_95=0.40, ap_small=0.20, recall=0.55, latency_ms=6.0, model_size_mb=7.0, verified_local=True),
        PaperParetoCandidate(candidate_id="paper-claim", recipe_id="r3", map50_95=0.50, verified_local=False),
        PaperParetoCandidate(candidate_id="inherited", recipe_id="r4", map50_95=0.60, verified_local=True, inheritance_depth=1),
        PaperParetoCandidate(candidate_id="slicing", recipe_id="r5", map50_95=0.70, verified_local=True, slicing_metrics={"sliced_map50_95": 0.70}),
    ])
    assert [item.candidate_id for item in report.included] == ["good"]
    assert report.dominated == ["dominated"]
    assert report.excluded["paper-claim"] == "local_verified_evidence_required"
    assert report.excluded["inherited"] == "current_node_only_evidence_required"
    assert report.excluded["slicing"] == "inference_policy_changed_not_training_pareto"


def test_single_seed_can_be_pareto_but_remains_possible() -> None:
    report = build_paper_pareto_report([PaperParetoCandidate(candidate_id="single", recipe_id="r1", map50_95=0.41, verified_local=True, evidence_status="possible")])
    assert report.included[0].evidence_status == "possible"
