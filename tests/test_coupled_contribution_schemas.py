import pytest

from yolo_agent.agents.coupled_contribution import (
    CoupledArmObservation,
    CoupledContributionReport,
)


def test_coupled_observation_requires_matched_protocol_identity() -> None:
    observation = CoupledArmObservation(
        arm="A",
        node_id="node-a",
        matched_control_node_id="control-seed-1",
        seed=1,
        protocol_hash="protocol-1",
        metric_deltas={"ap_small": 0.01},
        paired_result_verified=True,
    )

    assert observation.evidence_role == "current_observation"
    with pytest.raises(ValueError, match="matched control"):
        CoupledArmObservation(
            arm="A",
            node_id="node-a",
            matched_control_node_id="",
            seed=1,
            protocol_hash="protocol-1",
            metric_deltas={"ap_small": 0.01},
        )


def test_coupled_contribution_report_yaml_round_trip(tmp_path) -> None:
    report = CoupledContributionReport(
        recipe_id="coupled-one",
        component_a="head.p2_small_object",
        component_b="sampling.small_object",
        complete_seeds=[1],
        incomplete_seeds={2: ["missing_arms:B"]},
    )

    output = report.to_yaml(tmp_path / "coupled_contribution_report.yaml")
    restored = CoupledContributionReport.from_yaml(output)

    assert restored == report
