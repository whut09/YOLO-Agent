"""CPU-only route tests for the eight domain adaptation strategies."""

from __future__ import annotations

import pytest
import torch

from yolo_agent.components.adapters.domain_adaptation.branch_runtime import (
    DomainAdaptationBranchPlugin,
    _require_strategy_assets,
)
from yolo_agent.components.adapters.domain_adaptation.domain_evidence import (
    DomainDatasetManifest,
    resolve_domain_protocol,
)
from yolo_agent.components.adapters.domain_adaptation.branches import DomainProtocolError
from yolo_agent.components.adapters.domain_adaptation.branches import default_domain_adaptation_registry


def _protocol() -> dict[str, object]:
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
    return resolve_domain_protocol(
        source=source,
        target=target,
        adaptation_mode="unsupervised",
    ).model_dump(mode="json")


def _options(branch_id: str) -> dict[str, object]:
    options: dict[str, object] = {
        "branch_id": branch_id,
        "weight": 0.1,
        "source_manifest": "source.yaml",
        "target_manifest": "target.yaml",
        "domain_protocol": _protocol(),
        "imgsz": 640,
        "runtime_strategy": default_domain_adaptation_registry().get(branch_id).runtime_strategy,
    }
    if branch_id == "pseudo_label_adaptation":
        options["pseudo_label_manifest"] = "pseudo.yaml"
    elif branch_id in {"domain_distillation", "cross_domain_teacher"}:
        options.update({"teacher_checkpoint": "teacher.pt", "teacher_sha256": "t" * 64})
    elif branch_id == "source_free_adaptation":
        options.update({
            "source_model_checkpoint": "source-model.pt",
            "source_model_sha256": "s" * 64,
        })
    elif branch_id == "contrastive_domain_alignment":
        options.update({"contrastive_pair_manifest": "pairs.yaml", "temperature": 0.1})
    elif branch_id == "active_domain_adaptation":
        options.update({"query_manifest": "queries.yaml", "label_budget": 2})
    return options


def _features() -> list[torch.Tensor]:
    return [torch.randn(4, 3, 2, 2, requires_grad=True)]


@pytest.mark.parametrize(
    "branch_id,asset_key",
    [
        ("pseudo_label_adaptation", "pseudo_label_manifest"),
        ("domain_distillation", "teacher_checkpoint"),
        ("cross_domain_teacher", "teacher_checkpoint"),
        ("source_free_adaptation", "source_model_checkpoint"),
        ("contrastive_domain_alignment", "contrastive_pair_manifest"),
        ("active_domain_adaptation", "query_manifest"),
    ],
)
def test_route_payload_asset_is_required(branch_id: str, asset_key: str) -> None:
    options = _options(branch_id)
    options.pop(asset_key)
    plugin = DomainAdaptationBranchPlugin(**options)
    with pytest.raises(DomainProtocolError, match="missing"):
        _require_strategy_assets(plugin.config.model_dump(mode="json"), plugin.config.runtime_strategy)


def test_each_non_teacher_route_consumes_its_batch_evidence() -> None:
    features = _features()
    domains = torch.tensor([0, 0, 1, 1])
    cases = [
        ("pseudo_label_adaptation", {"pseudo_labels": torch.tensor([0.2, 0.4])}),
        (
            "contrastive_domain_alignment",
            {"contrastive_pairs": torch.randn(2, 2, 3)},
        ),
        ("active_domain_adaptation", {"query_ids": torch.tensor([1, 0])}),
    ]
    for branch_id, batch in cases:
        plugin = DomainAdaptationBranchPlugin(**_options(branch_id))
        loss = plugin._compute_runtime_strategy_loss(features, domains, batch=batch, device=features[0].device)
        assert torch.isfinite(loss)
        loss.backward()
        assert features[0].grad is not None
        features[0].grad.zero_()


@pytest.mark.parametrize("branch_id", ["domain_distillation", "cross_domain_teacher"])
def test_teacher_routes_consume_teacher_features(branch_id: str) -> None:
    features = _features()
    domains = torch.tensor([0, 0, 1, 1])
    teacher_features = torch.randn(2, 3)
    plugin = DomainAdaptationBranchPlugin(**_options(branch_id))
    loss = plugin._compute_runtime_strategy_loss(
        features,
        domains,
        batch={"teacher_features": teacher_features},
        device=features[0].device,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert features[0].grad is not None


def test_teacher_route_uses_explicit_target_domain_id() -> None:
    features = [torch.randn(4, 3, 2, 2, requires_grad=True)]
    domains = torch.tensor([2, 2, 7, 7])
    options = _options("domain_distillation")
    options.update({"source_domain_id": 2, "target_domain_id": 7})
    plugin = DomainAdaptationBranchPlugin(**options)
    loss = plugin._compute_runtime_strategy_loss(
        features,
        domains,
        batch={"teacher_features": torch.randn(2, 3)},
        device=features[0].device,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert features[0].grad is not None


def test_teacher_route_fails_without_batch_or_local_reference() -> None:
    plugin = DomainAdaptationBranchPlugin(**_options("domain_distillation"))
    with pytest.raises(DomainProtocolError, match="missing"):
        plugin._compute_runtime_strategy_loss(
            _features(),
            torch.tensor([0, 0, 1, 1]),
            batch={},
            device=torch.device("cpu"),
        )


def test_existing_reference_asset_hash_is_verified(tmp_path) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    options = _options("domain_distillation")
    options["teacher_checkpoint"] = str(checkpoint)
    options["teacher_sha256"] = "0" * 64
    with pytest.raises(DomainProtocolError, match="sha256 mismatch"):
        _require_strategy_assets(options, "domain_teacher_distillation")


def test_adversarial_route_attaches_trainable_discriminator() -> None:
    plugin = DomainAdaptationBranchPlugin(**_options("adversarial_alignment"))
    model = torch.nn.Linear(2, 2)
    plugin.build_model(context=None, trainer=None, model=model)
    assert plugin._domain_discriminator is not None
    loss = plugin.compute_loss(_features(), torch.tensor([0, 0, 1, 1]))
    assert torch.isfinite(loss)


def test_cpu_route_does_not_create_production_maturity_evidence() -> None:
    plugin = DomainAdaptationBranchPlugin(**_options("feature_alignment"))
    assert plugin.evidence is None
    loss = plugin.compute_loss(_features(), torch.tensor([0, 0, 1, 1]))
    assert torch.isfinite(loss)
    assert plugin.evidence is None
