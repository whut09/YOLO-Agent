from __future__ import annotations

from yolo_agent.research.mechanism_cluster_markdown import (
    render_mechanism_cluster_markdown,
)
from yolo_agent.research.mechanism_clusters import (
    AdapterCoverageOpportunity,
    MechanismClusterConflict,
    MechanismClusterSummary,
    PaperMechanismClusterReport,
)


def test_markdown_separates_mapping_from_runtime_implementation() -> None:
    report = PaperMechanismClusterReport(
        paper_count=3,
        matched_paper_count=2,
        unresolved_paper_count=1,
        clusters=[MechanismClusterSummary(
            cluster_id="attention_blocks",
            adapter_family="model_graph.attention",
            training_semantic="feature_attention_transformation",
            paper_ids=["a", "b"],
            paper_count=2,
        )],
        conflicts=[MechanismClusterConflict(
            paper_id="c",
            candidate_cluster_ids=["feature_distillation", "logits_distillation"],
            reason="ambiguous_training_semantics",
        )],
        implementation_opportunities=[AdapterCoverageOpportunity(
            rank=1,
            cluster_id="attention_blocks",
            adapter_family="model_graph.attention",
            paper_ids=["a", "b"],
            paper_count=2,
            runtime_hooks=["build_model"],
            implementation_status="adapter_required",
            score=198.0,
        )],
    ).with_hash()

    rendered = render_mechanism_cluster_markdown(report)

    assert "Mechanism mapping is not runtime implementation" in rendered
    assert "Covers 2 papers through `model_graph.attention`" in rendered
    assert "ambiguous_training_semantics" in rendered
    assert "paper claim" not in rendered.lower()
