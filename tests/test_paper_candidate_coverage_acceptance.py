from __future__ import annotations

from pathlib import Path

from yolo_agent.agents.paper_proposal_ledger import (
    PaperCandidateCoverageLedger,
    planned_recipe_disposition,
)


RUNTIME_READY_COMPONENTS = [
    "loss.hard_negative_classification",
    "sampling.hard_negative_replay",
    "loss.quality.correlation",
    "loss.quality.pseudo_iou",
    "assigner.task_aligned",
    "assigner.optimal_transport",
    "distillation.yolo26_teacher_student",
    "neck.rtmdet_large_kernel",
]


def test_evidence_bound_ten_proposals_are_never_dropped(tmp_path: Path) -> None:
    ledger = PaperCandidateCoverageLedger(
        tmp_path / "paper_candidate_coverage.yaml",
        run_id="coverage-acceptance",
        protocol_hash="coco-yolo26-640",
    )
    records = []
    for index, component_id in enumerate(RUNTIME_READY_COMPONENTS):
        records.append(
            planned_recipe_disposition(
                run_id="coverage-acceptance",
                round_index=1,
                recipe_id=f"paper.atomic.{index}",
                recipe_version="v1.0.0",
                component_ids=[component_id],
                decision="deferred" if index >= 6 else "selected",
                reasons=([] if index < 6 else ["pilot_budget_rank_deferred"]),
                related_papers=[f"paper-{index}"],
                method_profile_ids=[f"profile-{index}"],
                execution_fingerprint=f"atomic-{index}",
                budget_rank=index + 1,
            )
        )

    records.extend(
        [
            planned_recipe_disposition(
                run_id="coverage-acceptance",
                round_index=1,
                recipe_id="paper.coupled.hard_negative",
                recipe_version="v1.0.0",
                component_ids=[
                    "loss.hard_negative_classification",
                    "sampling.hard_negative_replay",
                ],
                combination_id="A+B",
                decision="deferred",
                reasons=["pilot_budget_rank_deferred"],
                related_papers=["paper-hard-negative-a", "paper-hard-negative-b"],
                method_profile_ids=["profile-hard-negative-a", "profile-hard-negative-b"],
                execution_fingerprint="coupled-hard-negative-a-b",
                budget_rank=9,
            ),
            planned_recipe_disposition(
                run_id="coverage-acceptance",
                round_index=1,
                recipe_id="paper.coupled.quality_assigner",
                recipe_version="v1.0.0",
                component_ids=[
                    "loss.quality.correlation",
                    "assigner.task_aligned",
                ],
                combination_id="A+B",
                decision="selected",
                reasons=[],
                related_papers=["paper-quality-assigner"],
                method_profile_ids=["profile-quality-assigner"],
                execution_fingerprint="coupled-quality-assigner-a-b",
                budget_rank=10,
            ),
        ]
    )

    coverage = ledger.upsert_many(records)

    assert len(coverage.records) == 10
    assert len(coverage.current_by_fingerprint) == 10
    assert {
        component_id
        for record in coverage.records
        for component_id in record.canonical_component_ids
    } >= set(RUNTIME_READY_COMPONENTS)
    assert coverage.disposition_counts == {
        "deferred_budget": 3,
        "queued": 7,
    }
    ledger.reconcile([record.execution_fingerprint for record in records])

