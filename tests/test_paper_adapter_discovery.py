from pathlib import Path

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
