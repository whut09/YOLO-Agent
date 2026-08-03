from pathlib import Path

import pytest
import yaml

from yolo_agent.certification.paper_adapter_discovery import (
    ReusablePaperAdapterDiscovery,
)


def _contracts(tmp_path: Path) -> Path:
    path = tmp_path / "components.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "components": {
                    "sampling.real": {
                        "display_name": "Real sampling",
                        "category": "sampling",
                        "implementation_path": (
                            "yolo_agent.components.adapters.sampling."
                            "small_object_sampling"
                        ),
                        "adapter_class": "SmallObjectSamplingAdapter",
                        "maturity": "adapter_implemented",
                    },
                    "fixture.dummy": {
                        "display_name": "Dummy fixture",
                        "category": "augmentation",
                        "implementation_path": "yolo_agent.components.adapters.dummy",
                        "adapter_class": "DummyAdapter",
                        "maturity": "adapter_implemented",
                    },
                    "broken.adapter": {
                        "display_name": "Broken adapter",
                        "category": "augmentation",
                        "implementation_path": "missing.adapter.module",
                        "adapter_class": "MissingAdapter",
                        "maturity": "adapter_implemented",
                    },
                    "paper.prior": {
                        "display_name": "Paper prior",
                        "category": "attention",
                        "maturity": "metadata_only",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_discovers_real_component_adapters_without_prefix_heuristics(
    tmp_path: Path,
) -> None:
    result = ReusablePaperAdapterDiscovery([_contracts(tmp_path)]).discover()

    assert [item.component_id for item in result.adapters] == ["sampling.real"]
    descriptor = result.adapters[0]
    assert descriptor.adapter_qualified_name.endswith(":SmallObjectSamplingAdapter")
    assert len(descriptor.identity.adapter_hash) == 64
    assert len(descriptor.identity.protocol_hash) == 64
    assert "broken.adapter" in result.errors
    assert "fixture.dummy" not in result.errors


def test_unrelated_code_commit_does_not_change_adapter_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "yolo_agent.certification.paper_adapter_discovery.current_code_commit",
        lambda: "commit-one",
    )
    first = ReusablePaperAdapterDiscovery([_contracts(tmp_path)]).discover().adapters[0]
    monkeypatch.setattr(
        "yolo_agent.certification.paper_adapter_discovery.current_code_commit",
        lambda: "commit-two",
    )
    second = ReusablePaperAdapterDiscovery([_contracts(tmp_path)]).discover().adapters[0]

    assert first.identity.code_commit != second.identity.code_commit
    assert first.identity.adapter_hash == second.identity.adapter_hash
    assert first.identity.protocol_hash == second.identity.protocol_hash
