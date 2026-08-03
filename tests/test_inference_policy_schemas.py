import pytest

from yolo_agent.certification.inference_policy_schemas import (
    InferencePolicyCertificationReport,
)
from yolo_agent.components.adapters.inference.policy import (
    InferencePolicyConfig,
    protocol_from_policy,
)


def _protocol():  # type: ignore[no-untyped-def]
    return protocol_from_policy(
        InferencePolicyConfig(
            policy_id="tta",
            kind="test_time_augmentation",
            scales=[1.0, 1.2],
        )
    )


def test_report_rejects_policy_metrics_in_standard_namespace() -> None:
    protocol = _protocol()
    with pytest.raises(ValueError, match="standard_640_metrics"):
        InferencePolicyCertificationReport(
            status="skipped",
            model="model.pt",
            annotations="instances.json",
            protocol=protocol,
            protocol_hash=protocol.protocol_hash,
            standard_640_metrics={"tta_map50_95": 0.5},
        )


def test_report_hash_binds_protocol() -> None:
    protocol = _protocol()
    report = InferencePolicyCertificationReport(
        status="skipped",
        model="model.pt",
        annotations="instances.json",
        protocol=protocol,
        protocol_hash=protocol.protocol_hash,
        reason="explicit execution required",
    )

    assert len(report.report_hash) == 64
    assert report.training_attribution_allowed is False
