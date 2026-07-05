"""COCO error fact selection tests."""

from __future__ import annotations

from yolo_agent.core.coco_error_selection import select_coco_error_facts
from yolo_agent.core.error_facts import ErrorFact


def test_coco_error_selection_focuses_baseline_unresolved_diagnoses() -> None:
    """Selector should choose high-signal baseline COCO errors for the next round."""
    facts = [
        _fact(
            fact_type="area_metric",
            subject="small",
            area="small",
            metric_name="ap_small",
            value=0.18,
            severity="high",
            actions=["small_object_recipe"],
        ),
        _fact(
            fact_type="per_class_metric",
            subject="bottle",
            class_name="bottle",
            metric_name="per_class_ar",
            value=0.22,
            severity="high",
            actions=["inspect_false_negatives", "class_balanced_sampling"],
        ),
        _fact(
            fact_type="localization_heavy_class",
            subject="person",
            class_name="person",
            count=40,
            severity="medium",
            actions=["bbox_loss_recipe"],
        ),
        _fact(
            candidate_id="candidate",
            node_id="node_candidate",
            fact_type="area_metric",
            subject="small",
            area="small",
            metric_name="ap_small",
            value=0.12,
            severity="high",
            actions=["should_not_be_selected"],
        ),
    ]

    selection = select_coco_error_facts(facts, max_focus=3)

    kinds = [item.diagnosis_kind for item in selection.current_round_focus]
    assert selection.baseline_node_ids == ["node_baseline"]
    assert "small_object_ap" in kinds
    assert "class_recall" in kinds
    assert "localization_class" in kinds
    assert "small_object_recipe" in selection.focus_action_candidates
    assert "should_not_be_selected" not in selection.focus_action_candidates


def test_coco_error_selection_warns_without_baseline_facts() -> None:
    """No baseline facts means the harness should not pretend to have targeted diagnoses."""
    selection = select_coco_error_facts(
        [
            _fact(
                candidate_id="candidate",
                node_id="node_candidate",
                fact_type="area_metric",
                subject="small",
                area="small",
                metric_name="ap_small",
                value=0.18,
                severity="high",
                actions=["small_object_recipe"],
            )
        ]
    )

    assert selection.current_round_focus == []
    assert selection.warnings == [
        "No baseline COCO error facts found; next-round proposals must not claim targeted learning."
    ]


def _fact(
    fact_type: str,
    subject: str,
    candidate_id: str = "yolo26s_coco_baseline",
    node_id: str = "node_baseline",
    class_name: str | None = None,
    area: str | None = None,
    metric_name: str | None = None,
    value: float | None = None,
    count: int | None = None,
    severity: str = "medium",
    actions: list[str] | None = None,
) -> ErrorFact:
    return ErrorFact(
        run_id="exp001",
        candidate_id=candidate_id,
        node_id=node_id,
        fact_type=fact_type,  # type: ignore[arg-type]
        subject=subject,
        class_name=class_name,
        area=area,
        metric_name=metric_name,
        value=value,
        count=count,
        severity=severity,  # type: ignore[arg-type]
        action_candidates=actions or [],
    )
