from __future__ import annotations

import pytest

pytest.importorskip("torch")

from yolo_agent.components.adapters.neck.common import YOLO26NeckConfig  # noqa: E402
from yolo_agent.components.adapters.neck.runtime import build_neck_component  # noqa: E402
from yolo_agent.components.graph_mechanisms import GRAPH_MECHANISMS  # noqa: E402


@pytest.mark.parametrize(
    "kind",
    [
        item.kind
        for item in GRAPH_MECHANISMS.values()
        if not item.requires_deformable_operator
    ],
)
def test_runtime_factory_builds_each_dependency_free_graph_mechanism(kind: str) -> None:
    spec = GRAPH_MECHANISMS[kind]
    config = YOLO26NeckConfig(kind=kind, component_id=spec.component_id)

    plugin = build_neck_component(kind, [16, 24, 32], config)

    assert plugin.plugin_id == spec.component_id
    assert plugin.input_contract.strides == [8, 16, 32]
    assert plugin.input_contract.channels == [16, 24, 32]
    assert plugin.output_contract == plugin.input_contract


def test_runtime_factory_builds_real_deformable_operator_when_available() -> None:
    pytest.importorskip("torchvision.ops")
    config = YOLO26NeckConfig(
        kind="deformable_feature_aggregation",
        component_id="neck.deformable_feature_aggregation",
        deformable_module="torchvision.ops",
    )

    plugin = build_neck_component(
        "deformable_feature_aggregation",
        [8, 16, 32],
        config,
    )

    assert plugin.plugin_id == "neck.deformable_feature_aggregation"
    assert plugin.operator_module == "torchvision.ops"
