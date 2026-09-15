"""Route-level behavior probes for distillation and domain-adaptation papers.

Each in-scope paper resolves to exactly one concrete route (branch-bound or
identity-recovery); these probes run the concrete mechanism behind the route
on synthetic CPU tensors and record the observed behavior: finite output,
backward without teacher gradients, mask semantics, EMA arithmetic, gradient
reversal direction, and pseudo-label filtering.  No training runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from yolo_agent.components.distillation.mechanism_losses import (
    DistillationInputs,
    build_distillation_mechanism_loss,
)
from yolo_agent.components.pseudo_label_filter import (
    PseudoLabelFilterError as PseudoLabelFilterErrorType,
)
from yolo_agent.components.pseudo_label_filter import filter_pseudo_labels
from yolo_agent.components.teacher_ema import TeacherEmaUpdater


@dataclass
class ProbeOutcome:
    """What one synthetic probe actually observed."""

    checks: dict[str, bool | str | int | float] = field(default_factory=dict)
    observed_changes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors and all(
            bool(value) for value in self.checks.values() if isinstance(value, bool)
        )


def _synthetic_distillation_inputs(mechanism: str) -> tuple[DistillationInputs, dict[str, torch.Tensor]]:
    """Build tensors that satisfy the mechanism's declared input contract."""

    student_inputs = {
        "student_logits": torch.randn(2, 4, 7, requires_grad=True),
        "teacher_logits": torch.randn(2, 4, 7),
    }
    watch: dict[str, torch.Tensor] = {"student_logits": student_inputs["student_logits"]}
    if mechanism in {"feature", "relation", "attention", "masked_feature", "contrastive", "cross_domain_teacher"}:
        student_inputs["student_features"] = [torch.randn(2, 5, 8, 8, requires_grad=True)]
        student_inputs["teacher_features"] = [torch.randn(2, 7, 8, 8)]
        watch["student_features"] = student_inputs["student_features"][0]
    if mechanism == "localization":
        student_inputs["student_boxes"] = torch.randn(2, 4, 7, requires_grad=True)
        student_inputs["teacher_boxes"] = torch.randn(2, 4, 7)
        watch["student_boxes"] = student_inputs["student_boxes"]
    if mechanism == "teacher_ensemble":
        student_inputs["teacher_logits"] = [
            torch.randn(2, 4, 7),
            torch.randn(2, 4, 7),
        ]
    return DistillationInputs(**student_inputs), watch


def probe_distillation_mechanism(mechanism: str) -> ProbeOutcome:
    """Run forward/backward for one distillation mechanism loss."""

    outcome = ProbeOutcome()
    try:
        inputs, watch = _synthetic_distillation_inputs(mechanism)
        options = {"class_dim": 1} if mechanism in {"logits", "quality_aware", "teacher_ensemble", "source_free_teacher", "cross_domain_teacher"} else {}
        output = build_distillation_mechanism_loss(mechanism, **options).compute(inputs)
        loss = output.loss
        outcome.checks["finite_output"] = bool(torch.isfinite(loss).item())
        loss.backward()
        grad_targets = {
            "logits": "student_logits",
            "quality_aware": "student_logits",
            "teacher_ensemble": "student_logits",
            "source_free_teacher": "student_logits",
            "cross_domain_teacher": "student_logits",
            "localization": "student_boxes",
        }
        watch_key = grad_targets.get(mechanism, "student_features")
        outcome.checks["backward_gradient_flows"] = watch[watch_key].grad is not None
        outcome.checks["finite_gradients"] = watch[watch_key].grad is not None and bool(
            torch.isfinite(watch[watch_key].grad).all().item()
        )
        outcome.checks["teacher_gradient_isolated"] = True
        for name, tensor in (
            ("teacher_logits", inputs.teacher_logits),
            ("teacher_features", inputs.teacher_features),
            ("teacher_boxes", inputs.teacher_boxes),
        ):
            if isinstance(tensor, (list, tuple)):
                if any(item.grad is not None for item in tensor if isinstance(item, torch.Tensor)):
                    outcome.checks["teacher_gradient_isolated"] = False
            elif isinstance(tensor, torch.Tensor) and tensor.grad is not None:
                outcome.checks["teacher_gradient_isolated"] = False
        outcome.observed_changes.append(
            f"{mechanism}:loss={float(loss.detach()):.6f},metrics={sorted(output.metrics)}"
        )
    except Exception as exc:  # noqa: BLE001 - probe boundary records everything
        outcome.errors.append(f"{type(exc).__name__}:{exc}")
    return outcome


def probe_teacher_ema(decay: float = 0.9) -> ProbeOutcome:
    """Prove the EMA teacher updates by the exact decay arithmetic."""

    outcome = ProbeOutcome()
    try:
        student = torch.nn.BatchNorm1d(4)
        with torch.no_grad():
            student.weight.fill_(1.0)
        updater = TeacherEmaUpdater(student, decay=decay)
        with torch.no_grad():
            student.weight.fill_(0.0)
        updater.update(student)
        expected = decay
        observed = float(updater.teacher.weight.flatten()[0])
        outcome.checks["ema_matches_decay"] = abs(observed - expected) < 1e-6
        outcome.checks["teacher_frozen"] = all(
            p.requires_grad is False for p in updater.teacher.parameters()
        )
        outcome.checks["teacher_eval_mode"] = updater.teacher.training is False
        student.num_batches_tracked.fill_(5)
        updater.update(student)
        outcome.checks["integer_buffer_copied"] = (
            int(updater.teacher.num_batches_tracked.item()) == 5
        )
        outcome.observed_changes.append(f"teacher_weight:{1.0:.4f}->{observed:.6f}")
    except Exception as exc:  # noqa: BLE001
        outcome.errors.append(f"{type(exc).__name__}:{exc}")
    return outcome


def probe_pseudo_label_filter(threshold: float = 0.5) -> ProbeOutcome:
    """Prove confidence filtering changes which labels reach the loss."""

    outcome = ProbeOutcome()
    try:
        labels = torch.tensor([1.0, 0.0, 2.0, 0.0])
        scores = torch.tensor([0.93, 0.30, 0.87, 0.20])
        result = filter_pseudo_labels(labels, scores, confidence_threshold=threshold)
        outcome.checks["kept_subset_smaller"] = result.kept_count < labels.numel()
        outcome.checks["kept_labels_match_scores"] = result.kept_labels.reshape(-1).tolist() == [
            1.0,
            2.0,
        ]
        labels2 = torch.tensor([1.0, 0.0, 2.0, 0.0])
        scores2 = torch.tensor([0.10, 0.30, 0.20, 0.20])
        try:
            filter_pseudo_labels(labels2, scores2, confidence_threshold=threshold)
            outcome.checks["empty_kept_set_fails_closed"] = False
        except PseudoLabelFilterErrorType:
            outcome.checks["empty_kept_set_fails_closed"] = True
        outcome.observed_changes.append(
            f"filter@{threshold}:kept={result.kept_count}/4"
        )
    except Exception as exc:  # noqa: BLE001
        outcome.errors.append(f"{type(exc).__name__}:{exc}")
    return outcome


def probe_gradient_reversal(scale: float = 1.0) -> ProbeOutcome:
    """Prove the reversal: forward passthrough, negated scaled backward."""

    from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
        _GradientReversal,
    )

    outcome = ProbeOutcome()
    try:
        features = torch.randn(4, 6, requires_grad=True)
        reversed_values = _GradientReversal.apply(features, scale)
        outcome.checks["forward_passthrough"] = bool(torch.equal(reversed_values, features))
        reversed_values.sum().backward()
        expected = -scale * torch.ones_like(features)
        outcome.checks["backward_negated"] = features.grad is not None and bool(
            torch.allclose(features.grad, expected)
        )
        outcome.observed_changes.append(f"grl_scale:{scale}:grad_sign=negated")
    except Exception as exc:  # noqa: BLE001
        outcome.errors.append(f"{type(exc).__name__}:{exc}")
    return outcome


__all__ = [
    "ProbeOutcome",
    "probe_distillation_mechanism",
    "probe_domain_branch_route",
    "probe_gradient_reversal",
    "probe_pseudo_label_filter",
    "probe_teacher_ema",
]


_STRATEGY_BATCH_EVIDENCE: dict[str, dict[str, torch.Tensor]] = {
    "target_pseudo_label_consistency": {
        "pseudo_labels": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        "pseudo_label_scores": torch.tensor([0.0, 0.0, 0.9, 0.0]),
    },
    "contrastive": {},
}


def probe_domain_branch_route(branch_id: str) -> ProbeOutcome:
    """Run one domain-adaptation branch strategy on synthetic CPU tensors."""

    from yolo_agent.components.adapters.domain_adaptation.branches import (
        default_domain_adaptation_registry,
    )
    from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
        DomainAdaptationBranchPlugin,
    )
    from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
        DomainDatasetManifest,
        resolve_domain_protocol,
    )

    outcome = ProbeOutcome()
    try:
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
        strategy = str(options["runtime_strategy"])
        if strategy == "target_pseudo_label_consistency":
            options["pseudo_label_manifest"] = "pseudo.yaml"
            options["confidence_threshold"] = 0.5
        elif strategy in {"domain_teacher_distillation", "cross_domain_teacher"}:
            options.update({"teacher_checkpoint": "teacher.pt", "teacher_sha256": "t" * 64})
        elif strategy == "source_free_target_adaptation":
            options.update(
                {
                    "source_model_checkpoint": "source-model.pt",
                    "source_model_sha256": "s" * 64,
                }
            )
        elif strategy == "cross_domain_contrastive":
            options.update({"contrastive_pair_manifest": "pairs.yaml", "temperature": 0.1})
        elif strategy == "active_query_selection":
            options.update({"query_manifest": "queries.yaml", "label_budget": 2})
        plugin = DomainAdaptationBranchPlugin(**options)
        features = [torch.randn(4, 3, 2, 2, requires_grad=True)]
        domains = torch.tensor([0, 0, 1, 1])
        batch: dict[str, torch.Tensor] = dict(_STRATEGY_BATCH_EVIDENCE.get(strategy, {}))
        if strategy == "cross_domain_contrastive":
            batch["contrastive_pairs"] = torch.randn(2, 2, 3)
        elif strategy == "active_query_selection":
            batch["query_ids"] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        elif strategy in {"domain_teacher_distillation", "cross_domain_teacher"}:
            batch["teacher_features"] = torch.randn(2, 3)
        elif strategy == "source_free_target_adaptation":
            batch["source_model_outputs"] = torch.randn(2, 3)
        if batch:
            loss = plugin._compute_runtime_strategy_loss(
                features, domains, batch=batch, device=features[0].device
            )
        else:
            loss = plugin.compute_loss(features, domains)
        outcome.checks["finite_output"] = bool(torch.isfinite(loss).item())
        loss.backward()
        outcome.checks["backward_gradient_flows"] = features[0].grad is not None and bool(
            torch.isfinite(features[0].grad).all().item()
        )
        outcome.observed_changes.append(
            f"{branch_id}:{strategy}:loss={float(loss.detach()):.6f}"
        )
    except Exception as exc:  # noqa: BLE001
        outcome.errors.append(f"{type(exc).__name__}:{exc}")
    return outcome
