"""Evidence-gated active assignment tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.assignment_fixtures import (
    detection_batch,
    native_model_and_criterion,
    run_one_shadow_batch,
    runtime_context,
    runtime_options,
)
from yolo_agent.components.adapters.assigners.yolo26_assignment import (
    AssignmentActivationGate,
    YOLO26AssignmentRuntimePlugin,
)


def test_active_assignment_requires_passed_shadow_evidence_and_replaces_only_o2m(
    tmp_path: Path,
) -> None:
    missing = AssignmentActivationGate().evaluate(
        tmp_path / "missing.json",
        component_id="assigner.task_aligned",
        method="tood_tal",
        assignment_path="one_to_many",
        minimum_batches=1,
        maximum_conflict_rate=1.0,
    )
    assert missing.allowed is False
    assert missing.blocked_by == ["shadow_evidence_missing"]

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    run_one_shadow_batch(shadow_dir, "tood_tal")
    shadow_path = shadow_dir / "assignment_tood_tal_shadow_evidence.json"
    model, criterion = native_model_and_criterion()
    native_o2m = criterion.one2many.assigner
    native_o2o = criterion.one2one.assigner
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    context = runtime_context(active_dir, "tood_tal")
    plugin = YOLO26AssignmentRuntimePlugin(
        **runtime_options(
            "tood_tal",
            mode="active",
            minimum_shadow_batches=1,
            shadow_evidence_path=str(shadow_path),
            shadow_payload_hash="payload-tood_tal-shadow",
        )
    )

    plugin.build_criterion(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
    )

    assert criterion.one2many.assigner is not native_o2m
    assert criterion.one2one.assigner is native_o2o
    image = torch.rand(1, 3, 64, 64)
    batch = detection_batch(image)
    predictions = model(image)
    output = criterion(predictions, batch)
    assert torch.isfinite(output[0]).all()
    returned = plugin.compute_loss(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
        predictions=predictions,
        batch=batch,
        loss_output=output,
    )
    assert returned is output
    evidence = json.loads(
        (active_dir / "assignment_tood_tal_active_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["assignment_path_replaced"] == "one_to_many"
    assert evidence["assignment_paths_replaced"] == ["one_to_many"]
    assert evidence["activation_source_sha256"]
    assert evidence["native_audit"]["dfl_free"] is True
    assert evidence["native_audit"]["nms_free"] is True


def test_dual_path_active_requires_and_executes_both_shadowed_paths(
    tmp_path: Path,
) -> None:
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    run_one_shadow_batch(shadow_dir, "dual_path", assignment_path="both")
    shadow_path = shadow_dir / "assignment_dual_path_shadow_evidence.json"
    model, criterion = native_model_and_criterion()
    native_o2m = criterion.one2many.assigner
    native_o2o = criterion.one2one.assigner
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    context = runtime_context(active_dir, "dual_path")
    plugin = YOLO26AssignmentRuntimePlugin(
        **runtime_options(
            "dual_path",
            mode="active",
            minimum_shadow_batches=1,
            shadow_evidence_path=str(shadow_path),
            shadow_payload_hash="payload-dual_path-shadow",
            assignment_path="both",
        )
    )

    plugin.build_criterion(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
    )

    assert criterion.one2many.assigner is not native_o2m
    assert criterion.one2one.assigner is not native_o2o
    image = torch.rand(1, 3, 64, 64)
    batch = detection_batch(image)
    predictions = model(image)
    output = criterion(predictions, batch)
    returned = plugin.compute_loss(
        context=context,
        trainer=SimpleNamespace(),
        model=model,
        criterion=criterion,
        predictions=predictions,
        batch=batch,
        loss_output=output,
    )

    assert returned is output
    evidence = json.loads(
        (active_dir / "assignment_dual_path_active_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["assignment_path_replaced"] == "both"
    assert evidence["assignment_paths_replaced"] == [
        "one_to_many",
        "one_to_one",
    ]
