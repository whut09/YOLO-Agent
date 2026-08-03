import pytest

from yolo_agent.agents.coupled_contribution import CoupledArmObservation


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
