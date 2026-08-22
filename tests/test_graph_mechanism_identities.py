from yolo_agent.components.graph_mechanisms import (
    GRAPH_COMPONENTS,
    GRAPH_MECHANISMS,
    P2_GRAPH_COMPONENT_ID,
    P2_GRAPH_STRIDES,
)


def test_graph_mechanisms_have_unique_runtime_identities() -> None:
    assert len(GRAPH_MECHANISMS) == 11
    assert len(GRAPH_COMPONENTS) == 11
    assert all(item.input_strides == (8, 16, 32) for item in GRAPH_COMPONENTS.values())
    assert all(item.output_strides == (8, 16, 32) for item in GRAPH_COMPONENTS.values())
    assert all(item.insertion_point == "before_detect" for item in GRAPH_COMPONENTS.values())


def test_deformable_dependency_and_p2_boundary_are_explicit() -> None:
    deformable = GRAPH_COMPONENTS["neck.deformable_feature_aggregation"]

    assert deformable.requires_deformable_operator is True
    assert P2_GRAPH_COMPONENT_ID not in GRAPH_COMPONENTS
    assert P2_GRAPH_STRIDES == (4, 8, 16, 32)
