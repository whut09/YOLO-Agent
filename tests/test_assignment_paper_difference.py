"""Baseline-versus-paper assignment difference tests.

Prompt-6 requires proof that a paper assigner produces a *different*
assignment than the baseline on constructed inputs.  The baseline here is the
pure-torch task-aligned reference (YOLOv8-family defaults) or an explicitly
injected native assignment, so these tests never depend on the local
ultralytics installation.  Provenance bindings are also checked so every
in-scope assignment mechanism points at the plugin method that implements it.
"""

from __future__ import annotations

import pytest
import torch

from tests.assignment_fixtures import assignment_inputs
from yolo_agent.components.assignment import (
    AssignerInputs,
    NativeYOLO26AssignerPlugin,
    build_yolo26_assigner_plugin,
    compare_assignments,
)
from yolo_agent.loss_math.formula_provenance import (
    ASSIGNMENT_FORMULA_PROVENANCE,
)
from yolo_agent.loss_math.tal_reference import TALReferenceParams, tal_reference_assignment

IN_SCOPE_ASSIGNERS = (
    "assigner.optimal_transport",
    "assigner.task_aligned",
    "assigner.dynamic_smooth_label",
)


def test_assignment_provenance_covers_every_in_scope_mechanism() -> None:
    for mechanism_id in IN_SCOPE_ASSIGNERS:
        provenance = ASSIGNMENT_FORMULA_PROVENANCE[mechanism_id]
        assert provenance.formula.strip()
        assert provenance.source_anchor.strip()
        assert provenance.parameter_semantics
        assert provenance.distinguishing_terms
        assert provenance.original_contract.strip()
        assert provenance.yolo26_contract.strip()
        assert provenance.adapter_transform.strip()
        assert provenance.information_preserved
        # Every in-scope mechanism is an adaptation onto the YOLO26 contract;
        # none of the original detectors is reproduced end to end.
        assert provenance.adaptation_class == "faithful_adaptation"
        assert provenance.information_approximated


def test_assignment_provenance_plugin_bindings_resolve() -> None:
    methods = {
        ASSIGNMENT_FORMULA_PROVENANCE[mechanism_id].implementation_method
        for mechanism_id in IN_SCOPE_ASSIGNERS
    }
    assert methods == {"ota", "tood_tal", "dsla"}
    for method in methods:
        assert build_yolo26_assigner_plugin(method) is not None


def test_ota_paper_assigner_differs_from_tal_reference_baseline() -> None:
    inputs = assignment_inputs()
    baseline = tal_reference_assignment(inputs, TALReferenceParams())
    candidate = build_yolo26_assigner_plugin("ota").run(inputs)

    comparison = compare_assignments(baseline, candidate)
    # The TAL baseline hands out top-k positives; OTA's dynamic supply for this
    # fixture gives one positive per GT, so the foreground sets must disagree.
    assert comparison.foreground_disagreement_count > 0
    assert comparison.candidate_positive_count != comparison.baseline_positive_count
    assert candidate.target_gt_indices.shape == baseline.target_gt_indices.shape


def test_ota_dynamic_supply_responds_to_iou_distribution() -> None:
    """OTA paper property: positive supply grows with the IoU distribution."""

    plugin = build_yolo26_assigner_plugin("ota")

    def inputs_with(scores_row0: torch.Tensor) -> AssignerInputs:
        base = assignment_inputs()
        scores = base.predicted_scores.clone()
        scores[0, : scores.shape[1], 0] = scores_row0[: scores.shape[1]]
        return AssignerInputs(**{**base.__dict__, "predicted_scores": scores})

    # Well-localized predictions (boxes equal to the 96x96 GT) versus poor ones.
    good = assignment_inputs()
    good_boxes = good.gt_boxes_xyxy.expand(1, 20, 4).contiguous()
    good_inputs = AssignerInputs(
        **{**good.__dict__, "predicted_boxes_xyxy": good_boxes}
    )
    bad_boxes = good.predicted_boxes_xyxy * 0.25
    bad_inputs = AssignerInputs(
        **{**good.__dict__, "predicted_boxes_xyxy": bad_boxes}
    )

    good_count = int(plugin.run(good_inputs).foreground_mask.sum())
    bad_count = int(plugin.run(bad_inputs).foreground_mask.sum())
    assert good_count > bad_count


def test_dsla_paper_assigner_differs_from_tal_reference_baseline() -> None:
    inputs = assignment_inputs()
    baseline = tal_reference_assignment(inputs, TALReferenceParams())
    candidate = build_yolo26_assigner_plugin("dsla").run(inputs)

    comparison = compare_assignments(baseline, candidate)
    # DSLA gates on interval score + centerness + online IoU while the TAL
    # reference selects top-k by alignment, so the constructed fixture must
    # produce a different positive set.
    assert comparison.foreground_disagreement_count > 0
    assert comparison.candidate_positive_count != comparison.baseline_positive_count
    # Dynamic smooth labels: the GT-class channel carries the continuous
    # quality value (never a binary one-hot), while other classes stay zero.
    positive_scores = candidate.target_scores[0][candidate.foreground_mask[0]]
    positive_labels = candidate.target_labels[0][candidate.foreground_mask[0]]
    positive_quality = positive_scores.gather(
        -1, positive_labels.unsqueeze(-1).long()
    ).squeeze(-1)
    assert float(positive_quality.max()) < 1.0
    assert float(positive_quality.min()) > 0.0


def test_dsla_stride_interval_gate_blocks_far_level_anchors() -> None:
    """DSLA paper property: with multiple stride levels, an anchor whose max
    side distance leaves its level's stride*8 interval receives zero scale
    score and cannot become positive, even though it lies inside the GT."""

    inputs = assignment_inputs()
    # Give the first ten diagonal points stride 8 and the rest stride 16 so
    # both interval levels are active.
    strides = torch.full((20, 1), 8.0)
    strides[10:, 0] = 16.0
    multi_stride = AssignerInputs(
        **{**inputs.__dict__, "stride_per_anchor": strides}
    )
    candidate = build_yolo26_assigner_plugin("dsla").run(multi_stride)

    chosen = inputs.anchor_points_xy[candidate.foreground_mask[0]]
    gt = inputs.gt_boxes_xyxy[0, 0]
    for point in chosen.tolist():
        inside = gt[0] < point[0] < gt[2] and gt[1] < point[1] < gt[3]
        max_distance = max(
            point[0] - gt[0], gt[2] - point[0], point[1] - gt[1], gt[3] - point[1]
        )
        if inside:
            # Stride-8 points must sit inside the relaxed 64*1.2 upper bound;
            # stride-16 points must clear the relaxed 64*0.8 lower bound.
            if point[0] < 80.0:
                assert max_distance <= 76.8
            else:
                assert max_distance >= 51.2
        else:
            assert not candidate.foreground_mask[0][
                int(((inputs.anchor_points_xy - point).abs().sum(dim=-1)).argmin())
            ]
    # The near-edge stride-8 point (4, 4): inside the GT, but max side
    # distance 92 > 76.8, so the interval prior must exclude it.
    near_edge = (inputs.anchor_points_xy - torch.tensor([4.0, 4.0])).abs().sum(dim=-1).argmin()
    assert not bool(candidate.foreground_mask[0][near_edge])


def test_dsla_zero_iou_prediction_cannot_become_positive() -> None:
    inputs = assignment_inputs()
    boxes = inputs.predicted_boxes_xyxy.clone()
    boxes[0, :, :] = 0.0  # degenerate predictions: IoU with the GT is 0
    zero_iou = AssignerInputs(**{**inputs.__dict__, "predicted_boxes_xyxy": boxes})

    candidate = build_yolo26_assigner_plugin("dsla").run(zero_iou)
    # Online IoU quality is multiplicative, so quality collapses to zero.
    assert not bool(candidate.foreground_mask.any())
    assert float(candidate.target_scores.abs().sum()) == 0.0


def test_pp_yoloee_tal_paper_assigner_differs_from_injected_native_baseline() -> None:
    try:
        import ultralytics  # noqa: F401
    except Exception:  # pragma: no cover - broken local cv2/numpy ABI
        pytest.skip("ultralytics unavailable in this environment")
    inputs = assignment_inputs()

    def native(*_: object) -> tuple[torch.Tensor, ...]:
        foreground = torch.zeros((1, 20), dtype=torch.bool)
        foreground[:, :2] = True
        labels = torch.zeros((1, 20), dtype=torch.long)
        boxes = torch.zeros((1, 20, 4))
        scores = torch.zeros((1, 20, 2))
        scores[:, :2, 0] = 1.0
        indices = torch.zeros((1, 20), dtype=torch.long)
        return labels, boxes, scores, foreground, indices

    baseline = NativeYOLO26AssignerPlugin(native).run(inputs)
    candidate = build_yolo26_assigner_plugin(
        "tood_tal", topk=13, alpha=1.0, beta=6.0
    ).run(inputs)

    comparison = compare_assignments(baseline, candidate)
    assert comparison.foreground_disagreement_count > 0
    assert comparison.candidate_positive_count != comparison.baseline_positive_count
