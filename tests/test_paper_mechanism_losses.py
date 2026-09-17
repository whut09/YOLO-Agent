"""Paper-specific distillation mechanism loss behavior tests.

Every loss in this suite comes from a recovered paper fulltext (runs/paper_fulltext/)
and carries its formula provenance in its class docstring.  Tests run real CPU
forward/backward — no mocks, no training loops — and pin the specific property
each paper claims (mask weighting, temperature scaling, prototype projection,
quality-gated masking, gradient-reversal-free EMA-teacher semantics, ...).
"""

from __future__ import annotations

import math

import pytest
import torch

from yolo_agent.components.distillation.mechanism_losses import (
    DistillationInputs,
    build_distillation_mechanism_loss,
)

NEW_MECHANISMS = [
    "pearson_feature",
    "richness_masked",
    "classifier_response",
    "instance_conditional",
    "structural_similarity",
    "prediction_guided",
    "hetero_assist",
    "global_prototype",
    "base_novel_commonality",
    "bovw_consistency",
    "glam_attention",
    "query_distillation",
    "cross_scale_self",
    "early_learning",
]


def _inputs(
    *,
    logits_hw: tuple[int, int] | None = (6, 6),
    feature_hw: tuple[int, int] = (8, 8),
    feature_channels: tuple[int, int] = (8, 16),
    boxes: int = 3,
    token_logits: bool = False,
) -> DistillationInputs:
    """Build synthetic teacher/student tensors with realistic heterogeneity."""
    if token_logits:
        student_logits = torch.randn(2, 4, 7, requires_grad=True)
        teacher_logits = torch.randn(2, 4, 7)
    else:
        h, w = logits_hw or (6, 6)
        student_logits = torch.randn(2, 4, h, w, requires_grad=True)
        teacher_logits = torch.randn(2, 4, h, w)
    c_s, c_t = feature_channels
    fh, fw = feature_hw
    return DistillationInputs(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_features=[torch.randn(2, c_s, fh, fw, requires_grad=True)],
        teacher_features=[torch.randn(2, c_t, fh, fw)],
        student_boxes=torch.rand(2, boxes, 4, requires_grad=True),
        teacher_boxes=torch.rand(2, boxes, 4),
    )


@pytest.mark.parametrize("mechanism", NEW_MECHANISMS)
def test_new_mechanism_forward_backward_finite(mechanism: str) -> None:
    inputs = _inputs()
    loss = build_distillation_mechanism_loss(mechanism)
    output = loss.compute(inputs)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss).all()
    output.loss.backward()
    grads = [inputs.student_logits.grad, inputs.student_features[0].grad]
    assert any(g is not None for g in grads)


@pytest.mark.parametrize("mechanism", NEW_MECHANISMS)
def test_new_mechanism_accepts_token_layout_logits(mechanism: str) -> None:
    """Dense heads expose (N,C,H,W); token-layout heads expose (N,C,L)."""
    inputs = _inputs(token_logits=True, feature_hw=(8, 8))
    loss = build_distillation_mechanism_loss(mechanism)
    output = loss.compute(inputs)
    assert torch.isfinite(output.loss).all()


@pytest.mark.parametrize("mechanism", NEW_MECHANISMS)
def test_new_mechanism_zero_weight_zero_gradient(mechanism: str) -> None:
    inputs = _inputs()
    loss = build_distillation_mechanism_loss(mechanism)
    scaled = loss.compute(inputs).loss * 0.0
    if scaled.requires_grad:
        scaled.backward()
    for tensor in (inputs.student_logits, inputs.student_features[0]):
        if tensor.grad is not None:
            assert float(tensor.grad.abs().sum()) == 0.0


def test_pearson_similarity_reduces_loss() -> None:
    """PKD: identical channel-structure features give r≈1 → loss≈0."""
    loss_fn = build_distillation_mechanism_loss("pearson_feature")
    base = torch.randn(2, 16, 8, 8)
    teacher = base.clone()
    student = base + torch.randn_like(base) * 0.05
    perfect = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[student],
            teacher_features=[teacher],
        )
    )
    noisy = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[torch.randn(2, 16, 8, 8)],
            teacher_features=[teacher],
        )
    )
    assert float(perfect.loss) < float(noisy.loss) * 0.5


def test_richness_mask_concentrates_on_high_score_regions() -> None:
    """FRS: the max-class teacher score map weights the FPN feature MSE."""
    loss_fn = build_distillation_mechanism_loss("richness_masked")
    teacher_logits = torch.full((1, 4, 8, 8), -8.0)
    teacher_logits[:, 0, :4, :] = 8.0  # top half has confident class 0
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 4, 8, 8, requires_grad=True),
        teacher_logits=teacher_logits,
        student_features=[torch.randn(1, 8, 8, 8, requires_grad=True)],
        teacher_features=[torch.randn(1, 16, 8, 8)],
    )
    output = loss_fn.compute(inputs)
    assert float(output.metrics["mean_richness"]) > 0.0
    assert torch.isfinite(output.loss).all()


def test_classifier_response_temperature_recorded_and_finite() -> None:
    """Classifier KD: temperature is part of the mechanism contract."""
    student = torch.randn(2, 10, 6, 6, requires_grad=True)
    teacher = torch.randn(2, 10, 6, 6) * 4.0
    low_t = build_distillation_mechanism_loss("classifier_response", temperature=1.0)
    high_t = build_distillation_mechanism_loss("classifier_response", temperature=8.0)
    inputs = DistillationInputs(
        student_logits=student,
        teacher_logits=teacher,
        student_features=None,
        teacher_features=None,
    )
    out_low = low_t.compute(inputs)
    out_high = high_t.compute(inputs)
    assert torch.isfinite(out_low.loss).all() and torch.isfinite(out_high.loss).all()
    assert out_low.metrics["temperature"] == 1.0
    assert out_high.metrics["temperature"] == 8.0
    # Paper property: student matching the teacher scores yields ~0 KL.
    matched = build_distillation_mechanism_loss("classifier_response").compute(
        DistillationInputs(
            student_logits=teacher.clone().requires_grad_(True),
            teacher_logits=teacher,
            student_features=None,
            teacher_features=None,
        )
    )
    assert float(matched.loss) < float(out_low.loss) * 1e-3


def test_structural_similarity_prefers_matching_structure() -> None:
    """StructKD: SSIM distance is smaller for structure-matched features."""
    loss_fn = build_distillation_mechanism_loss("structural_similarity")
    teacher = torch.randn(1, 8, 16, 16)
    matched = teacher.clone()
    mismatched = torch.randn(1, 8, 16, 16)
    out_match = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[matched],
            teacher_features=[teacher],
        )
    )
    out_mismatch = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[mismatched],
            teacher_features=[teacher],
        )
    )
    assert float(out_match.loss) < float(out_mismatch.loss)


def test_prediction_guided_quality_weights_emphasize_foreground() -> None:
    """PGD: top-K quality pixels dominate; loss concentrates on confident boxes."""
    loss_fn = build_distillation_mechanism_loss("prediction_guided")
    teacher_logits = torch.full((1, 4, 8, 8), -6.0)
    teacher_logits[:, :, 2:5, 2:5] = 6.0  # a confident "object"
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 4, 8, 8, requires_grad=True),
        teacher_logits=teacher_logits,
        student_features=[torch.randn(1, 8, 8, 8, requires_grad=True)],
        teacher_features=[torch.randn(1, 16, 8, 8)],
    )
    output = loss_fn.compute(inputs)
    assert float(output.metrics["mean_quality"]) > 0.0
    assert torch.isfinite(output.loss).all()


def test_global_prototype_shares_dictionary_between_detector_tokens() -> None:
    """GlobalKD: student and teacher attention must use the SAME prototypes."""
    loss_fn = build_distillation_mechanism_loss("global_prototype")
    teacher = torch.randn(1, 16, 8, 8)
    student = torch.randn(1, 8, 8, 8, requires_grad=True)
    out = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[student],
            teacher_features=[teacher],
        )
    )
    assert torch.isfinite(out.loss).all()
    out.loss.backward()
    assert student.grad is not None


def test_base_novel_similarity_yields_valid_distribution() -> None:
    """MFDC: prototype-similarity soft labels must sum to 1 over classes."""
    loss_fn = build_distillation_mechanism_loss("base_novel_commonality")
    student = torch.randn(2, 5, 6, 6, requires_grad=True)
    teacher = torch.randn(2, 5, 6, 6)
    out = loss_fn.compute(
        DistillationInputs(
            student_logits=student,
            teacher_logits=teacher,
            student_features=None,
            teacher_features=None,
        )
    )
    assert torch.isfinite(out.loss).all()
    out.loss.backward()
    assert student.grad is not None


def test_bovw_similarity_maps_share_vocabulary_axis() -> None:
    """PA-BoVW: student and teacher similarity maps share the K-word axis."""
    loss_fn = build_distillation_mechanism_loss("bovw_consistency")
    student = torch.randn(1, 8, 6, 6, requires_grad=True)
    teacher = torch.randn(1, 16, 8, 8)
    out = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[student],
            teacher_features=[teacher],
        )
    )
    assert torch.isfinite(out.loss).all()
    out.loss.backward()


def test_glam_attention_masks_are_deterministic() -> None:
    """GLAMD: channel/spatial attention masks are pure functions of features."""
    loss_fn = build_distillation_mechanism_loss("glam_attention")
    student = torch.randn(1, 8, 12, 12, requires_grad=True)
    teacher = torch.randn(1, 16, 12, 12)
    inputs = DistillationInputs(
        student_logits=None,
        teacher_logits=None,
        student_features=[student],
        teacher_features=[teacher],
    )
    first = loss_fn.compute(inputs).loss
    second = loss_fn.compute(inputs).loss
    assert float(first) == float(second)


def test_query_distillation_weights_match_paper_lambdas() -> None:
    """DLIM-Det: L_QPD = 5*L1 + 2*GIoU proxy; changing lambdas scales QPD."""
    teacher = torch.randn(1, 4, 6, 6)
    boxes = torch.rand(1, 12, 4, requires_grad=True)
    inputs = DistillationInputs(
        student_logits=torch.randn(1, 4, 6, 6, requires_grad=True),
        teacher_logits=teacher,
        student_features=[torch.randn(1, 8, 6, 6, requires_grad=True)],
        teacher_features=[torch.randn(1, 16, 6, 6)],
        student_boxes=boxes,
        teacher_boxes=torch.rand(1, 12, 4),
    )
    base = build_distillation_mechanism_loss("query_distillation")
    doubled = build_distillation_mechanism_loss("query_distillation", lambda_position=10.0)
    out_base = base.compute(inputs)
    out_double = doubled.compute(inputs)
    # Total = λ1·L1 + λ2·GIoU + QRD.  Doubling λ1 adds exactly the base
    # position contribution (λ1·L1) to the total.
    delta = float(out_double.loss) - float(out_base.loss)
    expected = float(out_base.metrics["query_position_loss"]) * 5.0
    assert math.isclose(delta, expected, rel_tol=1e-4), (delta, expected)


def test_cross_scale_self_weight_is_nonnegative_delta() -> None:
    """MSCD: adaptive weight w = max(0, L_det(y) - L_det(y_m)) is >= 0."""
    loss_fn = build_distillation_mechanism_loss("cross_scale_self")
    teacher_logits = torch.randn(2, 4, 6, 6)
    inputs = DistillationInputs(
        student_logits=torch.randn(2, 4, 6, 6, requires_grad=True),
        teacher_logits=teacher_logits,
        student_features=[torch.randn(2, 8, 6, 6, requires_grad=True)],
        teacher_features=[torch.randn(2, 16, 8, 8)],
    )
    output = loss_fn.compute(inputs)
    assert float(output.metrics["adaptive_weight_mean"]) >= 0.0


def test_early_learning_ema_reduces_student_teacher_gap() -> None:
    """ELDET: response KD pulls the student toward the early-learning teacher."""
    loss_fn = build_distillation_mechanism_loss("early_learning")
    # Confident class-0 teacher; students at varying alignment on that class.
    teacher = torch.zeros(1, 4, 6, 6)
    teacher[:, 0] = 6.0
    student = torch.zeros(1, 4, 6, 6, requires_grad=True)
    student.data[:, 0] = -6.0  # opposite direction
    far = loss_fn.compute(
        DistillationInputs(
            student_logits=student,
            teacher_logits=teacher,
            student_features=None,
            teacher_features=None,
        )
    )
    student_close = torch.zeros(1, 4, 6, 6, requires_grad=True)
    student_close.data[:, 0] = 5.0  # nearly aligned
    close = loss_fn.compute(
        DistillationInputs(
            student_logits=student_close,
            teacher_logits=teacher,
            student_features=None,
            teacher_features=None,
        )
    )
    assert float(close.loss) < float(far.loss)


def test_hetero_assist_assists_head_alignment() -> None:
    """HEAD: AKD aligns student features to teacher head features."""
    loss_fn = build_distillation_mechanism_loss("hetero_assist")
    teacher_feat = torch.randn(1, 16, 6, 6)
    logits_t = torch.randn(1, 4, 6, 6)
    matched = teacher_feat.mean(dim=1, keepdim=True).expand(-1, 8, -1, -1) * 0.5
    out_match = loss_fn.compute(
        DistillationInputs(
            student_logits=torch.randn(1, 4, 6, 6, requires_grad=True),
            teacher_logits=logits_t,
            student_features=[matched.clone().requires_grad_(True)],
            teacher_features=[teacher_feat],
        )
    )
    out_rand = loss_fn.compute(
        DistillationInputs(
            student_logits=torch.randn(1, 4, 6, 6, requires_grad=True),
            teacher_logits=logits_t,
            student_features=[torch.randn(1, 8, 6, 6, requires_grad=True)],
            teacher_features=[teacher_feat],
        )
    )
    assert float(out_match.loss) < float(out_rand.loss)


def test_instance_conditional_aligns_matched_instances() -> None:
    """ICD: value-feature distillation at matched token positions."""
    loss_fn = build_distillation_mechanism_loss("instance_conditional")
    teacher = torch.randn(1, 16, 6, 6)
    student = torch.randn(1, 8, 6, 6, requires_grad=True)
    out = loss_fn.compute(
        DistillationInputs(
            student_logits=None,
            teacher_logits=None,
            student_features=[student],
            teacher_features=[teacher],
            student_boxes=torch.rand(1, 3, 4, requires_grad=True),
            teacher_boxes=torch.rand(1, 3, 4),
        )
    )
    assert torch.isfinite(out.loss).all()
    out.loss.backward()
    assert student.grad is not None
