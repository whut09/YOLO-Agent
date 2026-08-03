from pathlib import Path

from yolo_agent.certification.component_schemas import (
    ComponentCertificationReport,
    ComponentCertificationStage,
    ComponentGPUCertificationEvidence,
    ComponentGPUProtocol,
    ComponentGPUResources,
)
from yolo_agent.certification.matched_pilot_fixture import (
    MatchedPilotCertificationFixture,
    MatchedPilotFixtureBuilder,
)
from yolo_agent.certification.paper_adapter_factory_schemas import (
    AdapterCertificationIdentity,
)


def test_gpu_report_materializes_non_claiming_matched_pilot_fixture(
    tmp_path: Path,
) -> None:
    identity = AdapterCertificationIdentity(
        component_id="sampling.small_object",
        adapter_hash="a" * 64,
        code_commit="commit-one",
        ultralytics_version="8.4.0",
        protocol_hash="protocol-one",
    )
    gpu_protocol = ComponentGPUProtocol(
        component_id=identity.component_id,
        adapter_hash=identity.adapter_hash,
        runtime_payload_hash="b" * 64,
        fixture_manifest_hash="f" * 64,
        model_sha256="m" * 64,
        ultralytics_version=identity.ultralytics_version,
        device="0",
    )
    evidence = ComponentGPUCertificationEvidence(
        component_id=identity.component_id,
        status="passed",
        worker_protocol_hash=identity.protocol_hash,
        gpu_protocol=gpu_protocol,
        runtime_payload_path=tmp_path / "runtime.yaml",
        runtime_payload_hash="b" * 64,
        checks={
            name: True
            for name in (
                "real_ultralytics_train",
                "required_hooks_observed",
                "backward_observed",
                "amp_enabled",
                "checkpoint_saved",
                "resume_completed",
                "resume_checkpoint_saved",
                "adapter_hash_matched",
                "fixture_manifest_matched",
                "adapter_artifacts_complete",
                "component_profile_verified",
                "stateful_resume_hook_observed",
            )
        },
        resources=ComponentGPUResources(
            device="0",
            gpu_name="Mock GPU",
            total_vram_mb=24000,
            peak_vram_mb=1000,
            train_duration_s=1,
            resume_duration_s=1,
            latency_ms=2,
            model_size_mb=5,
        ),
    )
    evidence_path = evidence.to_yaml(tmp_path / "gpu_evidence.yaml")
    report = ComponentCertificationReport(
        component_id=identity.component_id,
        mode="gpu",
        status="passed",
        initial_maturity="smoke_passed",
        final_maturity="gpu_certified",
        next_maturity="pilot_reproduced",
        protocol_hash=identity.protocol_hash,
        adapter_hash=identity.adapter_hash,
        code_commit=identity.code_commit,
        ultralytics_version=identity.ultralytics_version,
        registry_path=tmp_path / "registry.yaml",
        workdir=tmp_path,
        stages=[
            ComponentCertificationStage(
                stage_id="cpu_smoke_precondition", status="passed"
            ),
            ComponentCertificationStage(stage_id="isolated_gpu_smoke", status="passed"),
        ],
        generated_paths={"gpu_evidence": evidence_path},
    )

    output = tmp_path / "matched_pilot_fixture.yaml"
    fixture = MatchedPilotFixtureBuilder().build(
        report=report,
        identity=identity,
        model="yolo26n.pt",
        data="mini-coco.yaml",
        output=output,
    )

    assert fixture.baseline.imgsz == fixture.candidate.imgsz == 640
    assert fixture.baseline.adapter_hash is None
    assert fixture.candidate.adapter_hash == identity.adapter_hash
    assert fixture.local_metric_claim_allowed is False
    assert fixture.maturity_ceiling == "gpu_certified"
    assert MatchedPilotCertificationFixture.from_yaml(output) == fixture
