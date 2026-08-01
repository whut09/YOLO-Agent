from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.certification.component_gpu_profiles import (
    validate_component_gpu_profile,
)
from yolo_agent.components.adapters import AdapterContext, AdapterRuntimePayload
from yolo_agent.components.adapters.registry import ComponentAdapterRegistry
from yolo_agent.components.adapters.losses.quality_alignment import (
    AuxiliaryLossEvidence,
    AuxiliaryPaperPrior,
)
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    DistillationEvidence,
)
from yolo_agent.components.adapters.head.p2_head import (
    P2HeadCheckpointReport,
    P2HeadManifest,
)
from yolo_agent.components.adapters.neck.common import YOLO26NeckManifest
from yolo_agent.components.model_graph import (
    ModelGraphResourceLimits,
    ModelGraphResourceReport,
    PartialCheckpointAudit,
)
from yolo_agent.components.adapters.sampling.small_object_sampling import (
    SmallObjectSamplingManifest,
)
from yolo_agent.components.contracts import load_contracts


def _sampling_payload(tmp_path: Path) -> AdapterRuntimePayload:
    contract = load_contracts(
        "configs/components/sampling/small_object_sampling.yaml"
    )[0]
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    payload = adapter.build_runtime_payload(
        AdapterContext(contract=contract, workspace=tmp_path),
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert payload is not None
    return payload


def test_sampling_gpu_profile_requires_bound_train_only_manifest(
    tmp_path: Path,
) -> None:
    payload = _sampling_payload(tmp_path)
    manifest = SmallObjectSamplingManifest(
        dataset_manifest="fixture-hash",
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        split="train",
        seed=17,
        area_thresholds={"small": 0.01},
        image_count=2,
        small_image_count=1,
        raw_weights=[2.0, 1.0],
        final_weights=[2.0, 1.0],
        image_paths=["a.png", "b.png"],
        clipping_statistics={"max_weight": 3.0},
        sample_count=2,
        adapter_hash="a" * 64,
        val_unchanged=True,
    )
    path = tmp_path / "sampler_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    checks = validate_component_gpu_profile(
        "sampling.small_object",
        payload,
        {"adapter_sampler_manifest": path},
    )

    assert all(value is True for value in checks.values())


def test_sampling_gpu_profile_rejects_unbound_manifest(tmp_path: Path) -> None:
    payload = _sampling_payload(tmp_path)
    path = tmp_path / "sampler_manifest.json"
    path.write_text(
        SmallObjectSamplingManifest(
            dataset_manifest="fixture-hash",
            protocol_hash="wrong",
            runtime_payload_hash=payload.payload_hash,
            split="train",
            seed=17,
            area_thresholds={"small": 0.01},
            image_count=1,
            small_image_count=1,
            raw_weights=[1.0],
            final_weights=[1.0],
            image_paths=["a.png"],
            clipping_statistics={"max_weight": 3.0},
            sample_count=1,
            adapter_hash="a" * 64,
        ).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling_protocol_bound"):
        validate_component_gpu_profile(
            "sampling.small_object",
            payload,
            {"adapter_sampler_manifest": path},
        )


def _loss_payload(tmp_path: Path, component_id: str) -> AdapterRuntimePayload:
    contract = next(
        item
        for item in load_contracts("configs/components/loss/quality_alignment.yaml")
        if item.component_id == component_id
    )
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    payload = adapter.build_runtime_payload(
        AdapterContext(contract=contract, workspace=tmp_path),
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "imgsz=640"],
        generated_config={},
    )
    assert payload is not None
    return payload


def test_correlation_gpu_profile_requires_real_loss_and_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    component_id = "loss.quality.correlation"
    payload = _loss_payload(tmp_path, component_id)
    metadata = tmp_path / "last.pt.auxiliary_loss.correlation.json"
    metadata.write_text("{}", encoding="utf-8")
    evidence = AuxiliaryLossEvidence(
        component_id=component_id,
        loss_name="correlation",
        changed_variable="loss.correlation.weight",
        weight=0.2,
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        adapter_version="1",
        plugin_version="1",
        plugin_sha256="a" * 64,
        rank=0,
        batch_log_name="aux/correlation",
        compute_loss_calls=1,
        latest_weighted_loss=0.1,
        total_loss_changed=True,
        native_assigner="native",
        native_bbox_loss="native_dfl_free",
        native_dfl_enabled=False,
        paper_prior=AuxiliaryPaperPrior(
            paper_id="paper",
            adaptation="component adaptation",
        ),
        checkpoint_metadata_paths=[str(metadata)],
    )
    evidence_path = tmp_path / "auxiliary_loss_correlation_evidence.json"
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")

    checks = validate_component_gpu_profile(
        component_id,
        payload,
        {"adapter_auxiliary_loss_correlation_evidence": evidence_path},
    )

    assert all(value is True for value in checks.values())


def test_bpc_gpu_profile_rejects_zero_runtime_contribution(tmp_path: Path) -> None:
    component_id = "loss.calibration.bpc"
    payload = _loss_payload(tmp_path, component_id)
    metadata = tmp_path / "last.pt.auxiliary_loss.bpc_calibration.json"
    metadata.write_text("{}", encoding="utf-8")
    evidence = AuxiliaryLossEvidence(
        component_id=component_id,
        loss_name="bpc_calibration",
        changed_variable="loss.bpc_calibration.weight",
        weight=0.1,
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        adapter_version="1",
        plugin_version="1",
        plugin_sha256="a" * 64,
        rank=0,
        batch_log_name="aux/bpc",
        compute_loss_calls=1,
        latest_weighted_loss=0.0,
        total_loss_changed=False,
        native_assigner="native",
        native_bbox_loss="native_dfl_free",
        native_dfl_enabled=False,
        paper_prior=AuxiliaryPaperPrior(
            paper_id="paper",
            adaptation="component adaptation",
        ),
        checkpoint_metadata_paths=[str(metadata)],
    )
    evidence_path = tmp_path / "auxiliary_loss_bpc_calibration_evidence.json"
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="loss_total_changed"):
        validate_component_gpu_profile(
            component_id,
            payload,
            {"adapter_auxiliary_loss_bpc_calibration_evidence": evidence_path},
        )


def test_pseudo_iou_gpu_profile_preserves_native_dfl_free_regression(
    tmp_path: Path,
) -> None:
    component_id = "loss.quality.pseudo_iou"
    payload = _loss_payload(tmp_path, component_id)
    metadata = tmp_path / "last.pt.auxiliary_loss.pseudo_iou.json"
    metadata.write_text("{}", encoding="utf-8")
    evidence = AuxiliaryLossEvidence(
        component_id=component_id,
        loss_name="pseudo_iou",
        changed_variable="loss.pseudo_iou.weight",
        weight=0.1,
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        adapter_version="1",
        plugin_version="1",
        plugin_sha256="a" * 64,
        rank=0,
        batch_log_name="aux/pseudo_iou",
        compute_loss_calls=2,
        latest_weighted_loss=0.05,
        total_loss_changed=True,
        native_assigner="native",
        native_bbox_loss="native_dfl_free",
        native_dfl_enabled=False,
        paper_prior=AuxiliaryPaperPrior(
            paper_id="paper",
            adaptation="component adaptation",
        ),
        checkpoint_metadata_paths=[str(metadata)],
    )
    evidence_path = tmp_path / "auxiliary_loss_pseudo_iou_evidence.json"
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")

    checks = validate_component_gpu_profile(
        component_id,
        payload,
        {"adapter_auxiliary_loss_pseudo_iou_evidence": evidence_path},
    )

    assert checks["loss_native_yolo26_preserved"] is True


def test_distillation_gpu_profile_requires_teacher_safety_and_resume(
    tmp_path: Path,
) -> None:
    component_id = "distillation.yolo26_teacher_student"
    contract = load_contracts(
        "configs/components/distillation/yolo26_teacher_student.yaml"
    )[0]
    teacher = tmp_path / "yolo26s.pt"
    student = tmp_path / "yolo26n.pt"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    payload = adapter.build_runtime_payload(
        AdapterContext(
            contract=contract,
            workspace=tmp_path,
            options={
                "teacher": str(teacher),
                "student": str(student),
                "teacher_data": "fixture.yaml",
                "student_data": "fixture.yaml",
            },
        ),
        protocol_hash="protocol-1",
        base_command=[
            "yolo",
            "detect",
            "train",
            f"model={student}",
            "data=fixture.yaml",
            "imgsz=640",
        ],
        generated_config={},
    )
    evidence = DistillationEvidence(
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        teacher_checkpoint=str(teacher),
        teacher_checkpoint_sha256="a" * 64,
        student_checkpoint=str(student),
        student_checkpoint_sha256="b" * 64,
        dataset="fixture.yaml",
        split="train",
        shared_batch_tensor=True,
        compute_loss_calls=2,
        latest_terms={"logits": 0.1},
        total_loss_changed=True,
        teacher_eval=True,
        teacher_frozen=True,
        teacher_no_grad=True,
        student_inference_graph_unchanged=True,
        resume_checkpoint=str(student),
        resume_checkpoint_sha256="c" * 64,
        resume_validated=True,
    )
    evidence_path = tmp_path / "distillation_evidence.json"
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")

    checks = validate_component_gpu_profile(
        component_id,
        payload,
        {"adapter_distillation_evidence": evidence_path},
    )

    assert all(value is True for value in checks.values())


def test_p2_gpu_profile_requires_real_stride_four_detection_path(
    tmp_path: Path,
) -> None:
    component_id = "head.p2_small_object"
    contract = load_contracts("configs/components/head/yolo26_p2_small_object.yaml")[0]
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    payload = adapter.build_runtime_payload(
        AdapterContext(contract=contract, workspace=tmp_path),
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "model=yolo26n.pt", "imgsz=640"],
        generated_config={},
    )
    resources = ModelGraphResourceReport(
        base_latency_ms=1.0,
        candidate_latency_ms=1.1,
        latency_regression=0.1,
        base_vram_estimate_mb=100,
        candidate_vram_estimate_mb=110,
        vram_regression=0.1,
        base_parameter_count=100,
        candidate_parameter_count=110,
        parameter_regression=0.1,
        base_model_size_mb=5.0,
        candidate_model_size_mb=5.5,
        model_size_regression=0.1,
        limits=ModelGraphResourceLimits(),
        checks={"latency": True},
        passed=True,
    )
    manifest = P2HeadManifest(
        adapter_version="1",
        plugin_version="1",
        adapter_hash="a" * 64,
        protocol_hash=payload.protocol_hash,
        runtime_payload_hash=payload.payload_hash,
        generated_model_yaml="generated.yaml",
        generated_yaml_sha256="b" * 64,
        actual_tensor_strides=[4, 8, 16, 32],
        detect_input_count=4,
        native_end2end=True,
        native_reg_max=1,
        dfl_disabled=True,
        graph_integrated=True,
        detection_head_integrated=True,
        native_loss_integrated=True,
        checkpoint_integrated=True,
        checkpoint=P2HeadCheckpointReport(
            policy="partial_load_new_head",
            loaded=True,
            partial=False,
            checkpoint_sha256="c" * 64,
            matched_keys=["model.0.weight"],
        ),
        checkpoint_history=[
            P2HeadCheckpointReport(
                policy="partial_load_new_head",
                loaded=True,
                partial=True,
                checkpoint_sha256="d" * 64,
                matched_keys=["model.0.weight"],
                newly_initialized_keys=["model.29.weight"],
            )
        ],
        base_parameter_count=100,
        p2_parameter_count=110,
        parameter_delta=10,
        base_model_size_mb=5.0,
        p2_model_size_mb=5.5,
        model_size_delta_mb=0.5,
        latency_audit_imgsz=64,
        base_latency_ms=1.0,
        p2_latency_ms=1.1,
        latency_delta_ms=0.1,
        latency_risk="low",
        resources=resources,
    )
    manifest_path = tmp_path / "p2_head_manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    yaml_path = tmp_path / "generated_yolo26_p2.yaml"
    yaml_path.write_text("# generated", encoding="utf-8")

    checks = validate_component_gpu_profile(
        component_id,
        payload,
        {
            "adapter_p2_head_manifest": manifest_path,
            "adapter_p2_model_yaml": yaml_path,
        },
    )

    assert checks["p2_stride_four_observed"] is True
    assert checks["p2_partial_checkpoint_audited"] is True


def _neck_payload(tmp_path: Path, component_id: str) -> AdapterRuntimePayload:
    contract = next(
        item
        for item in load_contracts("configs/components/neck/yolo26_multi_scale.yaml")
        if item.component_id == component_id
    )
    adapter = ComponentAdapterRegistry().create_for_contract(contract)
    payload = adapter.build_runtime_payload(
        AdapterContext(contract=contract, workspace=tmp_path),
        protocol_hash="protocol-1",
        base_command=["yolo", "detect", "train", "model=yolo26n.pt", "imgsz=640"],
        generated_config={},
    )
    assert payload is not None
    return payload


@pytest.mark.parametrize(
    ("component_id", "neck_kind", "adapter_class"),
    [
        ("neck.multi_scale_fusion", "multi_scale_fusion", "MultiScaleFusionAdapter"),
        (
            "neck.gold_gather_distribute",
            "gold_gather_distribute",
            "GoldGatherDistributeAdapter",
        ),
        (
            "neck.rtmdet_large_kernel",
            "rtmdet_large_kernel",
            "RTMDetLargeKernelNeckAdapter",
        ),
    ],
)
def test_neck_gpu_profiles_require_graph_and_resource_evidence(
    tmp_path: Path,
    component_id: str,
    neck_kind: str,
    adapter_class: str,
) -> None:
    payload = _neck_payload(tmp_path, component_id)
    adapter_hash = str(payload.model_graph_plugin[0].options["adapter_hash"])
    resources = ModelGraphResourceReport(
        base_latency_ms=1.0,
        candidate_latency_ms=1.1,
        latency_regression=0.1,
        base_vram_estimate_mb=100,
        candidate_vram_estimate_mb=110,
        vram_regression=0.1,
        base_parameter_count=100,
        candidate_parameter_count=110,
        parameter_regression=0.1,
        base_model_size_mb=5.0,
        candidate_model_size_mb=5.5,
        model_size_regression=0.1,
        limits=ModelGraphResourceLimits(),
        checks={"all": True},
        passed=True,
    )
    manifest = YOLO26NeckManifest(
        component_id=component_id,
        neck_kind=neck_kind,
        adapter_class=adapter_class,
        adapter_version="1",
        plugin_class="YOLO26NeckRuntimePlugin",
        plugin_version="1",
        adapter_hash=adapter_hash,
        protocol_hash=payload.protocol_hash,
        insertion_point="before_detect",
        input_strides=[8, 16, 32],
        input_channels=[64, 128, 256],
        output_strides=[8, 16, 32],
        output_channels=[64, 128, 256],
        native_end2end=True,
        native_reg_max=1,
        dfl_disabled=True,
        checkpoint=PartialCheckpointAudit(
            loaded=True,
            partial=True,
            checkpoint_sha256="a" * 64,
            matched_keys=["model.0.weight"],
            newly_initialized_keys=["neck.weight"],
        ),
        resources=resources,
        export_dry_run=True,
    )
    path = tmp_path / f"{component_id.replace('.', '_')}_manifest.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    checks = validate_component_gpu_profile(
        component_id,
        payload,
        {f"adapter_{component_id}_manifest": path},
    )

    assert all(value is True for value in checks.values())
