"""Shadow assignment runtime equivalence and evidence tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.assignment_fixtures import (
    detection_batch,
    native_model_and_criterion,
    runtime_context,
    runtime_options,
)
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    AssignmentShadowEvidence,
    YOLO26AssignmentRuntimePlugin,
)


@pytest.mark.parametrize("method", ["tood_tal", "ota", "dsla"])
def test_real_yolo26_shadow_mode_preserves_native_loss_and_writes_evidence(
    method: str,
    tmp_path: Path,
) -> None:
    model, criterion = native_model_and_criterion()
    native_one_to_many = criterion.one2many.assigner
    native_one_to_one = criterion.one2one.assigner
    context = runtime_context(tmp_path, method)
    plugin = YOLO26AssignmentRuntimePlugin(
        **runtime_options(method, mode="shadow", minimum_shadow_batches=1)
    )
    assert plugin.build_criterion(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
    ) is criterion
    image = torch.rand(1, 3, 64, 64)
    batch = detection_batch(image)
    predictions = model(image)
    native_output = criterion(predictions, batch)
    native_loss = native_output[0].detach().clone()
    native_items = native_output[1].detach().clone()

    returned = plugin.compute_loss(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
        predictions=predictions,
        batch=batch,
        loss_output=native_output,
    )

    assert returned is native_output
    assert torch.equal(returned[0], native_loss)
    assert torch.equal(returned[1], native_items)
    assert criterion.one2many.assigner is native_one_to_many
    assert criterion.one2one.assigner is native_one_to_one
    evidence_path = tmp_path / f"assignment_{method}_shadow_evidence.json"
    evidence = AssignmentShadowEvidence.model_validate_json(
        evidence_path.read_text(encoding="utf-8")
    )
    assert evidence.aggregate.batches == 1
    assert evidence.aggregate.total_candidates > 0
    assert evidence.aggregate.baseline_positive_count > 0
    assert evidence.aggregate.candidate_positive_count > 0
    assert 0.0 <= evidence.aggregate.baseline_positive_ratio <= 1.0
    assert 0.0 <= evidence.aggregate.candidate_positive_ratio <= 1.0
    assert 0.0 <= evidence.aggregate.conflict_rate <= 1.0
    assert 0.0 <= evidence.aggregate.gt_conflict_rate <= 1.0
    assert 0.0 <= evidence.aggregate.matching_stability <= 1.0
    assert evidence.shadow_passed is True
    assert evidence.runtime_payload_hash == f"payload-{method}-shadow"
    assert evidence.changed_variables == context.payload.changed_variables
    assert evidence.assignment_path_replaced is None
    assert evidence.paper_prior.evidence_level == "paper_prior"
    assert evidence.paper_prior.reported_delta == {}
