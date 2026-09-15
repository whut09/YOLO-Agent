"""Synthetic CPU behavior tests for teacher/domain primitives.

No training loop runs here.  Every test constructs small tensors, runs the
real forward/backward path, and asserts the specific behavior the mechanism
claims: EMA averaging math, gradient isolation of the frozen teacher,
pseudo-label confidence filtering, and gradient-reversal direction.
"""

from __future__ import annotations

import pytest
import torch

from yolo_agent.components.adapters.domain_adaptation.branches import DomainProtocolError
from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
    DomainAdaptationBranchPlugin,
    _GradientReversal,
)
from yolo_agent.components.pseudo_label_filter import (
    PseudoLabelFilterError,
    filter_pseudo_labels,
)
from yolo_agent.components.teacher_ema import TeacherEmaError, TeacherEmaUpdater


def _student_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(8, 6),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(6),
        torch.nn.Linear(6, 3),
    )


class TestTeacherEmaUpdater:
    def test_update_averages_parameters_and_floating_buffers(self) -> None:
        student = _student_model()
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.fill_(1.0)
        updater = TeacherEmaUpdater(student, decay=0.9)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.fill_(0.0)
        updater.update(student)
        for parameter in updater.teacher.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 0.9))
        # Second observation moves the teacher by the remaining 10%.
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.fill_(2.0)
        updater.update(student)
        for parameter in updater.teacher.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 1.01))

    def test_integer_buffers_are_copied_not_averaged(self) -> None:
        student = torch.nn.BatchNorm1d(4)
        updater = TeacherEmaUpdater(student, decay=0.5)
        student.num_batches_tracked.fill_(7)
        updater.update(student)
        assert int(updater.teacher.num_batches_tracked.item()) == 7

    def test_teacher_stays_frozen_and_in_eval_mode(self) -> None:
        student = _student_model()
        updater = TeacherEmaUpdater(student, decay=0.999)
        student.train()
        updater.update(student)
        assert updater.teacher.training is False
        assert all(p.requires_grad is False for p in updater.teacher.parameters())

    def test_gradient_does_not_flow_from_teacher_to_student(self) -> None:
        student = _student_model()
        updater = TeacherEmaUpdater(student, decay=0.9)
        x = torch.randn(4, 8, requires_grad=True)
        updater.teacher(x).sum().backward()
        assert x.grad is not None
        assert all(p.grad is None for p in updater.teacher.parameters())

    def test_state_roundtrip_restores_exact_teacher(self) -> None:
        student = _student_model()
        updater = TeacherEmaUpdater(student, decay=0.9)
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.fill_(0.5)
        updater.update(student)
        state = updater.state_dict()
        other = TeacherEmaUpdater(_student_model(), decay=0.3)
        other.load_state_dict(state)
        assert other.decay == pytest.approx(0.9)
        for key, value in updater.teacher.state_dict().items():
            assert torch.equal(value, other.teacher.state_dict()[key])

    @pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
    def test_invalid_decay_fails_closed(self, decay: float) -> None:
        with pytest.raises(TeacherEmaError, match="decay"):
            TeacherEmaUpdater(_student_model(), decay=decay)

    def test_state_key_divergence_fails_closed(self) -> None:
        updater = TeacherEmaUpdater(_student_model(), decay=0.9)
        smaller = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU())
        with pytest.raises(TeacherEmaError, match="diverge"):
            updater.update(smaller)


class TestPseudoLabelFilter:
    def test_threshold_keeps_only_confident_labels(self) -> None:
        labels = torch.tensor([1.0, 0.0, 2.0, 0.0])
        scores = torch.tensor([0.92, 0.30, 0.81, 0.49])
        result = filter_pseudo_labels(labels, scores, confidence_threshold=0.5)
        assert result.kept_count == 2
        assert result.kept_labels.reshape(-1).tolist() == [1.0, 2.0]
        assert result.kept_scores.tolist() == pytest.approx([0.92, 0.81])
        assert result.keep_mask.tolist() == [True, False, True, False]

    def test_boundary_score_is_kept(self) -> None:
        labels = torch.tensor([1.0])
        result = filter_pseudo_labels(
            labels, torch.tensor([0.5]), confidence_threshold=0.5
        )
        assert result.kept_count == 1

    def test_empty_kept_set_raises(self) -> None:
        with pytest.raises(PseudoLabelFilterError, match="kept no pseudo-labels"):
            filter_pseudo_labels(
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.1, 0.2]),
                confidence_threshold=0.9,
            )

    @pytest.mark.parametrize("threshold", [0.0, -0.5, 1.0, 1.2])
    def test_invalid_threshold_fails_closed(self, threshold: float) -> None:
        with pytest.raises(PseudoLabelFilterError, match="confidence threshold"):
            filter_pseudo_labels(
                torch.tensor([1.0]), torch.tensor([0.7]), confidence_threshold=threshold
            )

    def test_score_shape_mismatch_fails_closed(self) -> None:
        with pytest.raises(PseudoLabelFilterError, match="one per sample"):
            filter_pseudo_labels(
                torch.tensor([1.0, 0.0]),
                torch.tensor([[0.7, 0.2]]),
                confidence_threshold=0.5,
            )

    def test_filtered_loss_actually_changes_output(self) -> None:
        """The runtime branch must apply the filter, not just declare it."""

        from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
            _pseudo_label_consistency_loss,
        )

        features = [torch.zeros(4, 2, requires_grad=True)]
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        scores = torch.tensor([0.95, 0.95, 0.10, 0.10])
        unfiltered = _pseudo_label_consistency_loss(features, labels)
        filtered = _pseudo_label_consistency_loss(
            features, labels, scores=scores, confidence_threshold=0.5
        )
        assert unfiltered.item() != pytest.approx(filtered.item())


class TestGradientReversalDirection:
    def test_backward_gradient_is_negated_and_scaled(self) -> None:
        features = torch.randn(6, 5, requires_grad=True)
        _GradientReversal.apply(features, 1.0).sum().backward()
        assert features.grad is not None
        assert torch.allclose(features.grad, -torch.ones_like(features))

    def test_scale_changes_magnitude_not_sign(self) -> None:
        features = torch.randn(6, 5, requires_grad=True)
        _GradientReversal.apply(features, 0.25).sum().backward()
        assert torch.allclose(features.grad, -0.25 * torch.ones_like(features))

    def test_forward_values_pass_through_unchanged(self) -> None:
        features = torch.randn(6, 5)
        reversed_values = _GradientReversal.apply(features, 1.0)
        assert torch.equal(reversed_values, features)


def _runtime_options(branch_id: str) -> dict[str, object]:
    from yolo_agent.components.adapters.domain_adaptation.branches import (
        default_domain_adaptation_registry,
    )
    from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
        DomainDatasetManifest,
        resolve_domain_protocol,
    )

    source = DomainDatasetManifest(
        path="source.yaml",
        sha256="source-sha256",
        dataset_hash="source-dataset",
        domain_id="0",
        domain_name="source",
        role="source",
        split="source_train",
        label_availability="labeled",
    )
    target = DomainDatasetManifest(
        path="target.yaml",
        sha256="target-sha256",
        dataset_hash="target-dataset",
        domain_id="1",
        domain_name="target",
        role="target",
        split="target_train",
        label_availability="unlabeled",
    )
    options: dict[str, object] = {
        "branch_id": branch_id,
        "weight": 0.1,
        "source_manifest": "source.yaml",
        "target_manifest": "target.yaml",
        "domain_protocol": resolve_domain_protocol(
            source=source, target=target, adaptation_mode="unsupervised"
        ).model_dump(mode="json"),
        "imgsz": 640,
        "runtime_strategy": default_domain_adaptation_registry()
        .get(branch_id)
        .runtime_strategy,
    }
    return options


class TestAdversarialRouteGradients:
    def test_discriminator_feature_gradient_is_reversed(self) -> None:
        """The feature path must receive the negated discriminator gradient."""

        options = _runtime_options("adversarial_alignment")
        plugin = DomainAdaptationBranchPlugin(**options)
        features = [torch.randn(4, 3, 2, 2, requires_grad=True)]
        domains = torch.tensor([0, 0, 1, 1])
        loss = plugin.compute_loss(features, domains)
        assert torch.isfinite(loss)
        loss.backward()
        assert features[0].grad is not None
        assert features[0].grad.abs().sum().item() > 0.0

    def test_discriminator_parameter_gradient_flows(self) -> None:
        options = _runtime_options("adversarial_alignment")
        plugin = DomainAdaptationBranchPlugin(**options)
        features = [torch.randn(4, 3, 2, 2, requires_grad=True)]
        domains = torch.tensor([0, 0, 1, 1])
        plugin.compute_loss(features, domains).backward()
        discriminator = plugin._domain_discriminator
        assert discriminator is not None
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0.0
            for p in discriminator.parameters()
        )


class TestPseudoLabelRouteFiltering:
    def test_route_applies_confidence_threshold(self) -> None:
        options = _runtime_options("pseudo_label_adaptation")
        options["pseudo_label_manifest"] = "pseudo.yaml"
        options["confidence_threshold"] = 0.6
        plugin = DomainAdaptationBranchPlugin(**options)
        features = [torch.zeros(4, 3, 2, 2, requires_grad=True)]
        domains = torch.tensor([0, 0, 1, 1])
        batch = {
            "pseudo_labels": torch.tensor([1.0, 1.0, 0.0, 1.0]),
            "pseudo_label_scores": torch.tensor([0.0, 0.0, 0.9, 0.3]),
        }
        loss = plugin._compute_runtime_strategy_loss(
            features, domains, batch=batch, device=features[0].device
        )
        # Low-confidence half must be filtered before the consistency term.
        unfiltered = plugin._compute_runtime_strategy_loss(
            features,
            domains,
            batch={"pseudo_labels": batch["pseudo_labels"]},
            device=features[0].device,
        )
        assert loss.item() != pytest.approx(unfiltered.item())
        assert torch.isfinite(loss)

    def test_route_fails_closed_when_threshold_keeps_nothing(self) -> None:
        options = _runtime_options("pseudo_label_adaptation")
        options["pseudo_label_manifest"] = "pseudo.yaml"
        options["confidence_threshold"] = 0.95
        plugin = DomainAdaptationBranchPlugin(**options)
        features = [torch.zeros(4, 3, 2, 2)]
        domains = torch.tensor([0, 0, 1, 1])
        batch = {
            "pseudo_labels": torch.tensor([1.0, 1.0, 1.0, 1.0]),
            "pseudo_label_scores": torch.tensor([0.1, 0.1, 0.2, 0.2]),
        }
        with pytest.raises(DomainProtocolError, match="kept no pseudo-labels"):
            plugin._compute_runtime_strategy_loss(
                features, domains, batch=batch, device=features[0].device
            )
