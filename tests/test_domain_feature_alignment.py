from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.domain_adaptation.feature_alignment import (
    CHANGED_VARIABLE,
    DomainFeatureAlignmentAdapter,
    DomainFeatureAlignmentRuntimePlugin,
    feature_statistics_alignment_loss,
)
from yolo_agent.components.contracts import ComponentContract
from yolo_agent.research.component_aliases import ComponentAliasResolver


def _contract() -> ComponentContract:
    return ComponentContract(
        component_id="domain_adaptation.general",
        display_name="Domain feature alignment",
        category="domain_adaptation",
        implementation_path=(
            "yolo_agent.components.adapters.domain_adaptation.feature_alignment"
        ),
        adapter_class="DomainFeatureAlignmentAdapter",
        insertion_point="trainer_loss",
        supported_detector_families=["yolo26"],
        supported_heads=["one_to_one", "one_to_many"],
        fixed_imgsz_compatible=True,
        maturity="adapter_implemented",
    )


def _context(tmp_path: Path, **options: object) -> AdapterContext:
    return AdapterContext(
        contract=_contract(),
        detector_family="yolo26",
        head="one_to_one",
        imgsz=640,
        workspace=tmp_path,
        options=dict(options),
    )


def _predictions() -> dict[str, object]:
    return {
        "one2many": {
            "feats": [
                torch.randn(4, 8, 4, 4, requires_grad=True),
                torch.randn(4, 16, 2, 2, requires_grad=True),
                torch.randn(4, 32, 1, 1, requires_grad=True),
            ]
        }
    }


def _runtime_context(payload: object, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        payload=payload,
        payload_path=tmp_path / "runtime_payload.yaml",
    )


def test_feature_alignment_shape_backward_amp() -> None:
    features = [
        torch.randn(4, 8, 4, 4, requires_grad=True),
        torch.randn(4, 16, 2, 2, requires_grad=True),
    ]
    domains = torch.tensor([0, 0, 1, 1])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = feature_statistics_alignment_loss(
            features,
            source_mask=domains == 0,
            target_mask=domains == 1,
            align_variance=True,
        )
    loss.backward()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert all(item.grad is not None for item in features)


def test_runtime_requires_explicit_source_and_target_domains(tmp_path: Path) -> None:
    adapter = DomainFeatureAlignmentAdapter()
    payload = adapter.build_runtime_payload(
        _context(tmp_path),
        protocol_hash="protocol",
        base_command=["yolo", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    plugin = DomainFeatureAlignmentRuntimePlugin(
        **payload.loss_plugin[0].options
    )
    native = (torch.ones(3, requires_grad=True), torch.ones(3))

    with pytest.raises(ValueError, match="refusing source-only fallback"):
        plugin.compute_loss(
            context=_runtime_context(payload, tmp_path),
            trainer=SimpleNamespace(),
            model=None,
            criterion=None,
            predictions=_predictions(),
            batch={},
            loss_output=native,
        )
    with pytest.raises(ValueError, match="source and target samples"):
        plugin.compute_loss(
            context=_runtime_context(payload, tmp_path),
            trainer=SimpleNamespace(),
            model=None,
            criterion=None,
            predictions=_predictions(),
            batch={"domain_id": torch.tensor([0, 0, 0, 0])},
            loss_output=native,
        )


def test_runtime_adds_loss_records_evidence_and_backpropagates(tmp_path: Path) -> None:
    adapter = DomainFeatureAlignmentAdapter()
    payload = adapter.build_runtime_payload(
        _context(tmp_path),
        protocol_hash="protocol",
        base_command=["yolo", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    payload.write(tmp_path / "runtime_payload.yaml")
    plugin = DomainFeatureAlignmentRuntimePlugin(**payload.loss_plugin[0].options)
    trainer = SimpleNamespace()
    predictions = _predictions()
    native_loss = torch.ones(3, requires_grad=True)

    updated, logged = plugin.compute_loss(
        context=_runtime_context(payload, tmp_path),
        trainer=trainer,
        model=None,
        criterion=None,
        predictions=predictions,
        batch={"domain_id": [0, 0, 1, 1]},
        loss_output=(native_loss, native_loss.detach()),
    )
    updated.sum().backward()

    evidence_path = tmp_path / "domain_feature_alignment_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert logged.numel() == 4
    assert trainer.auxiliary_loss_terms["domain_feature_alignment"] > 0
    assert evidence["source_samples"] == 2
    assert evidence["target_samples"] == 2
    assert evidence["runtime_payload_hash"] == payload.payload_hash
    assert evidence["exact_reproduction"] is False
    assert any(
        feature.grad is not None
        for feature in predictions["one2many"]["feats"]  # type: ignore[index]
    )


def test_zero_weight_is_native_equivalent_without_domain_metadata(
    tmp_path: Path,
) -> None:
    adapter = DomainFeatureAlignmentAdapter()
    context = _context(tmp_path, **{CHANGED_VARIABLE: 0.0})
    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["yolo", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    plugin = DomainFeatureAlignmentRuntimePlugin(**payload.loss_plugin[0].options)
    native_loss = torch.ones(3, requires_grad=True)

    updated, logged = plugin.compute_loss(
        context=_runtime_context(payload, tmp_path),
        trainer=SimpleNamespace(),
        model=None,
        criterion=None,
        predictions={},
        batch={},
        loss_output=(native_loss, native_loss.detach()),
    )

    assert torch.equal(updated, native_loss)
    assert logged[-1].item() == 0.0


def test_adapter_payload_and_smoke_are_runtime_bound(tmp_path: Path) -> None:
    adapter = DomainFeatureAlignmentAdapter()
    context = _context(tmp_path)

    payload = adapter.build_runtime_payload(
        context,
        protocol_hash="protocol",
        base_command=["yolo", "train", "imgsz=640"],
        generated_config={"imgsz": 640},
    )
    smoke = adapter.smoke_test(context)

    payload.verify_imports()
    assert payload.changed_variables == {CHANGED_VARIABLE: 0.05}
    assert payload.loss_plugin[0].required_hooks == ["compute_loss"]
    assert payload.supports_amp and payload.supports_ddp and payload.supports_resume
    assert smoke.passed and smoke.evidence_kind == "local"
    assert smoke.checks["explicit_source_target_batch"] is True


def test_domain_mechanism_resolves_to_registered_component_adaptation() -> None:
    resolver = ComponentAliasResolver.from_yaml()
    mapping = resolver.resolve("domain_adaptation").mappings[0]
    contract = resolver.contracts["domain_adaptation.general"]

    assert mapping.adapter_verified is True
    assert mapping.verified_adapter_ids == ["domain_adaptation.general"]
    assert mapping.artifact_execution_ready is False
    assert contract.maturity == "adapter_implemented"
    assert contract.changes_model_graph is False
