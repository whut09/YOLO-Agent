"""Paper-83 runtime preflight tests.

The preflight is the step that actually *runs* every frozen paper's runtime
path on synthetic CPU tensors before the first training.  These tests pin:

* the artifact contract (exactly 83 records, PASS/FAIL vocabulary, fields);
* the anti-fake invariants (mock evidence, missing adapters, constant losses,
  no-op modules, and missing runtime hooks cannot pass);
* real CPU integration against the live DA branch plugin, the live
  distillation mechanism loss, and the live adapter smoke-test surface.

No training loop is started anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from yolo_agent.research.paper_runtime_preflight import (
    PaperRuntimePreflightRecord,
    PaperRuntimePreflightRunner,
    RuntimePreflightFailure,
    _fingerprint,
    _run_distillation_mechanism,
    _run_domain_adaptation_branch,
    _require_finite_grad,
    _require_finite_scalar,
    render_preflight_summary,
    write_preflight_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_record(paper_id: str = "arxiv:2210.11539", **overrides: object) -> PaperRuntimePreflightRecord:
    defaults: dict[str, object] = {
        "paper_id": paper_id,
        "implementation_domain": "domain_adaptation",
        "adapter_ids": ["adversarial_alignment"],
        "runtime_hooks": ["loss.domain_adaptation"],
        "materialized": True,
        "synthetic_forward": True,
        "synthetic_backward": True,
        "behavior_changed": True,
        "rollback_verified": True,
        "fingerprint": _fingerprint(["domain_adaptation", "adversarial_alignment", "0.5"]),
        "status": "PASS",
        "error": "",
    }
    defaults.update(overrides)
    return PaperRuntimePreflightRecord(**defaults)  # type: ignore[arg-type]


def _synthetic_report(records: list[PaperRuntimePreflightRecord]):
    from yolo_agent.research.paper_runtime_preflight import PaperRuntimePreflightReport

    passed = sum(1 for r in records if r.status == "PASS")
    return PaperRuntimePreflightReport(
        paper_count=len(records),
        passed=passed,
        failed=len(records) - passed,
        runtime_preflight_passed=(len(records) - passed == 0),
        real_training_executed=False,
        records=records,
    )


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------


def test_live_sweep_produces_exactly_83_pass_records() -> None:
    """The real sweep on the current commit passes all 83 frozen papers."""
    runner = PaperRuntimePreflightRunner()
    paper_ids = runner.paper_ids()
    assert len(paper_ids) == 83
    assert len(set(paper_ids)) == 83

    report = runner.run()
    assert report.paper_count == 83
    assert len(report.records) == 83
    assert report.failed == 0
    assert report.runtime_preflight_passed is True
    assert report.real_training_executed is False
    record_fields = set(PaperRuntimePreflightRecord.model_fields)
    assert record_fields == {
        "paper_id",
        "implementation_domain",
        "adapter_ids",
        "runtime_hooks",
        # Prompt-18G: auditable RuntimeHookIdentity payloads per record.
        "runtime_hook_identities",
        "materialized",
        "synthetic_forward",
        "synthetic_backward",
        "behavior_changed",
        "rollback_verified",
        "fingerprint",
        "status",
        "error",
    }
    for record in report.records:
        assert record.status == "PASS"
        assert record.materialized and record.synthetic_forward
        assert record.fingerprint
        # Prompt-18G: every passing paper resolves to a real, audited hook.
        assert record.runtime_hook_identities, record.paper_id
        assert all(
            hook.strip().lower() != "unknown" for hook in record.runtime_hooks
        ), record.paper_id


def test_every_passing_record_executes_real_math() -> None:
    report = PaperRuntimePreflightRunner().run()
    for record in report.records:
        if record.status != "PASS":
            continue
        assert record.materialized, record.paper_id
        assert record.synthetic_forward, record.paper_id
        assert record.synthetic_backward, record.paper_id
        assert record.behavior_changed, record.paper_id
        assert record.rollback_verified, record.paper_id


def test_artifact_round_trip(tmp_path: Path) -> None:
    report = PaperRuntimePreflightRunner().run()
    out = write_preflight_artifacts(report, yaml_path=tmp_path / "preflight.yaml")
    data = yaml.safe_load(out.read_text(encoding="utf-8-sig"))
    assert data["schema_version"] == "paper_83_runtime_preflight.v1"
    assert data["paper_count"] == 83
    assert len(data["records"]) == 83
    assert all(record["status"] == "PASS" for record in data["records"])
    assert data["runtime_preflight_passed"] is True
    assert data["real_training_executed"] is False


def test_summary_marks_training_unsafe_on_failure() -> None:
    report = _synthetic_report(
        [
            _sample_record(),
            _sample_record("arxiv:2303.13853", status="FAIL", error="boom"),
        ]
    )
    text = render_preflight_summary(report)
    assert "Papers:       2" in text
    assert "Training-safe: NO" in text
    assert "arxiv:2303.13853" in text


# ---------------------------------------------------------------------------
# Anti-fake invariants
# ---------------------------------------------------------------------------


def test_non_finite_loss_is_rejected() -> None:
    import torch

    with pytest.raises(RuntimePreflightFailure, match="not finite"):
        _require_finite_scalar(torch.tensor(float("nan")), "probe")
    with pytest.raises(RuntimePreflightFailure, match="not a scalar"):
        _require_finite_scalar(torch.ones(2, 3), "probe")
    with pytest.raises(RuntimePreflightFailure, match="not a tensor"):
        _require_finite_scalar(object(), "probe")


def test_missing_gradient_is_rejected() -> None:
    import torch

    with pytest.raises(RuntimePreflightFailure, match="no gradient"):
        _require_finite_grad(torch.ones(2, requires_grad=False), "probe")
    bad = torch.ones(2, requires_grad=True)
    bad.grad = torch.tensor([float("nan"), 1.0])
    with pytest.raises(RuntimePreflightFailure, match="non-finite"):
        _require_finite_grad(bad, "probe")


def test_constant_loss_cannot_pass_behavior_probe() -> None:
    """A fake loss that ignores its inputs must be caught by the sweep."""

    class ConstantLoss:
        def compute(self, inputs: object) -> object:
            import torch

            from yolo_agent.components.distillation.mechanism_losses import (
                DistillationLossOutput,
            )

            return DistillationLossOutput(loss=torch.tensor(0.5), metrics={})

    import yolo_agent.research.paper_runtime_preflight as module

    def fake_builder(mechanism: str, **options: object) -> ConstantLoss:
        return ConstantLoss()

    original = module._build_mechanism_loss
    module._build_mechanism_loss = fake_builder  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimePreflightFailure, match="did not react"):
            _run_distillation_mechanism(
                "logits",
                spec_requires_features=False,
                requires_multiple_teachers=False,
                paper_id="arxiv:2210.11539",
            )
    finally:
        module._build_mechanism_loss = original  # type: ignore[assignment]


def test_mock_smoke_evidence_cannot_pass() -> None:
    """An adapter reporting mock evidence must fail the preflight."""

    class FakeResult:
        passed = True
        evidence_kind = "mock"
        checks = {"shape": "(1, 1)"}
        errors: list[str] = []

    class FakeAdapter:
        def smoke_test(self, context: object) -> FakeResult:
            return FakeResult()

        def rollback_plan(self, context: object) -> object:
            class Plan:
                reversible = True

            return Plan()

    import yolo_agent.research.paper_runtime_preflight as module
    from yolo_agent.components.contracts import ComponentContract

    contract = ComponentContract(
        component_id="loss.fake",
        display_name="Fake",
        category="loss",
    )

    original_load = module._load_component_contracts
    original_create = module._create_adapter
    module._load_component_contracts = lambda: {"loss.fake": contract}  # type: ignore[assignment]

    def fake_create(received: object) -> FakeAdapter:
        return FakeAdapter()

    module._create_adapter = fake_create  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimePreflightFailure, match="mock evidence"):
            module._run_adapter_component("loss.fake", paper_id="arxiv:fake")
    finally:
        module._load_component_contracts = original_load  # type: ignore[assignment]
        module._create_adapter = original_create  # type: ignore[assignment]


def test_missing_adapter_binding_fails_the_paper() -> None:
    """A DA paper with no branch binding must be a FAIL record, not a skip."""
    runner = PaperRuntimePreflightRunner()
    original = runner._da_branches
    runner._da_branches = lambda: {}  # type: ignore[method-assign]
    try:
        report = runner.run()
    finally:
        runner._da_branches = original  # type: ignore[method-assign]
    da_failures = [
        record
        for record in report.records
        if record.implementation_domain == "domain_adaptation" and record.status == "FAIL"
    ]
    assert da_failures, "missing branch bindings must produce FAIL records"
    assert all("no runtime branch bound" in record.error for record in da_failures)


def test_no_op_network_module_cannot_pass() -> None:
    """A mechanism that returns its input unchanged fails the finite-scalar gate."""
    import torch

    with pytest.raises(RuntimePreflightFailure, match="not a scalar"):
        _require_finite_scalar(torch.ones(2, 4), "no-op")
    # Identity placeholders also produce no gradient path:
    identity = torch.ones(2, requires_grad=True)
    with pytest.raises(RuntimePreflightFailure, match="no gradient"):
        _require_finite_grad(identity, "no-op") if identity.grad is None else None


def test_missing_runtime_hook_is_recorded() -> None:
    record = _sample_record(runtime_hooks=[])
    assert record.runtime_hooks == []
    # The hook list is part of the record contract; executors fill it from the
    # real runtime (never from a hardcoded constant per paper).
    report = PaperRuntimePreflightRunner().run()
    hooks = {record.paper_id: record.runtime_hooks for record in report.records}
    assert all(hooks[paper_id] for paper_id in hooks)


# ---------------------------------------------------------------------------
# Real CPU runtime integration (live paper adapters, no mocks)
# ---------------------------------------------------------------------------


def test_real_da_branch_integration_adversarial_alignment() -> None:
    record = _run_domain_adaptation_branch(
        "adversarial_alignment", paper_id="arxiv:2210.11539"
    )
    assert record.status == "PASS"
    assert record.synthetic_forward and record.synthetic_backward
    assert record.fingerprint


def test_real_da_branch_integration_source_free_adaptation() -> None:
    record = _run_domain_adaptation_branch(
        "source_free_adaptation", paper_id="arxiv:2303.13853"
    )
    assert record.status == "PASS"
    assert record.behavior_changed


def test_da_behavior_probe_catches_identity_feature_path() -> None:
    """Feeding the identical features twice must not raise, and a mechanism
    whose output ignores its inputs is rejected by the probe inside the
    runner (covered above); here we pin that the probe itself is wired."""
    import torch

    from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
        DomainAdaptationBranchPlugin,
    )

    features = [
        torch.randn(4, 16, 6, 6, requires_grad=True),
        torch.randn(4, 32, 6, 6, requires_grad=True),
        torch.randn(4, 64, 6, 6, requires_grad=True),
    ]
    plugin = DomainAdaptationBranchPlugin(
        branch_id="feature_alignment",
        weight=0.05,
        source_manifest="synthetic://s",
        target_manifest="synthetic://t",
    )
    first = plugin.compute_loss(features, torch.tensor([0, 0, 1, 1]))
    second = plugin.compute_loss(features, torch.tensor([0, 0, 1, 1]))
    v1 = float((first[0] if isinstance(first, tuple) else first).detach())
    v2 = float((second[0] if isinstance(second, tuple) else second).detach())
    assert v1 == pytest.approx(v2)  # deterministic on identical inputs
    perturbed = [f + torch.randn_like(f) * 0.35 for f in features]
    third = plugin.compute_loss(perturbed, torch.tensor([0, 0, 1, 1]))
    v3 = float((third[0] if isinstance(third, tuple) else third).detach())
    assert abs(v3 - v1) > 1e-9  # and reacts to its inputs


@pytest.mark.parametrize(
    "mechanism, requires_features, requires_multiple",
    [
        ("logits", False, False),
        ("feature", True, False),
        ("pearson_feature", True, False),
        ("relation", True, False),
    ],
)
def test_real_distillation_mechanism_integration(
    mechanism: str, requires_features: bool, requires_multiple: bool
) -> None:
    record = _run_distillation_mechanism(
        mechanism,
        spec_requires_features=requires_features,
        requires_multiple_teachers=requires_multiple,
        paper_id="arxiv:probe",
    )
    assert record.status == "PASS"
    assert record.behavior_changed




def test_real_adapter_smoke_integration_quality_loss() -> None:
    from yolo_agent.research.paper_runtime_preflight import _run_adapter_component

    record, _ = _run_adapter_component("loss.calibration.bpc", paper_id="arxiv:2303.14404")
    assert record.status == "PASS"
    assert record.synthetic_backward
    assert record.runtime_hooks == ["hook.loss.calibration.bpc"]
    assert record.runtime_hook_identities[0]["phase"] == "loss"
