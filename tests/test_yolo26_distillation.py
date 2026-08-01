import copy
import json
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest
import torch
import yaml

from yolo_agent.adapters.ultralytics.plugin_bridge import PluginCriterionWrapper
from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation.yolo26_distillation import (
    DistillationEvidence,
    YOLO26DistillationAdapter,
    YOLO26DistillationConfig,
    YOLO26DistillationRuntimePlugin,
)
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.distillation import (
    DistillationBatch,
    DistillationTrainerHook,
    DistillationWeights,
    MockDistillationTrainer,
    YOLO26DistillationLoss,
    distillation_loss,
)
from yolo_agent.recipes.schemas import recipe_from_mapping


def _context(tmp_path: Path, **updates) -> AdapterContext:
    options = {"teacher": "yolo26s.pt", "student": "yolo26n.pt", "teacher_data": "coco.yaml", "student_data": "coco.yaml", "teacher_split": "train", "student_split": "train", "imgsz": 640, "amp": True, "resume": False}
    options.update(updates)
    contract = ComponentContract(component_id="distillation.yolo26_teacher_student", display_name="Distillation", category="distillation", implementation_path="yolo_agent.components.adapters.distillation.yolo26_distillation", adapter_class="YOLO26DistillationAdapter", maturity="smoke_passed", fixed_imgsz_compatible=True)
    return AdapterContext(contract=contract, detector_family="yolo26", imgsz=640, workspace=tmp_path, options=options)


def test_distillation_shapes_and_backward() -> None:
    student_logits = torch.randn(2, 8, requires_grad=True)
    teacher_logits = torch.randn(2, 8, requires_grad=True)
    student_features = torch.randn(2, 4, 5, requires_grad=True)
    teacher_features = torch.randn(2, 4, 5, requires_grad=True)
    student_boxes = torch.randn(2, 6, 4, requires_grad=True)
    teacher_boxes = torch.randn(2, 6, 4, requires_grad=True)
    terms = distillation_loss(student_logits, teacher_logits, student_features=student_features, teacher_features=teacher_features, student_boxes=student_boxes, teacher_boxes=teacher_boxes)
    terms["total"].backward()
    assert student_logits.grad is not None and student_features.grad is not None and student_boxes.grad is not None
    assert teacher_logits.grad is None and teacher_features.grad is None and teacher_boxes.grad is None


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        distillation_loss(torch.randn(2, 8), torch.randn(3, 8))


def test_three_distillation_terms_have_independent_configurable_weights() -> None:
    terms = distillation_loss(
        torch.randn(2, 3, 8),
        torch.randn(2, 3, 8),
        student_features=[torch.randn(2, 4, 3, 3)],
        teacher_features=[torch.randn(2, 8, 3, 3)],
        student_boxes=torch.randn(2, 4, 8),
        teacher_boxes=torch.randn(2, 4, 8),
        weights=DistillationWeights(logits=0.5, feature=2.0, localization=3.0),
        logits_dim=1,
    )

    expected = 0.5 * terms["logits"] + 2.0 * terms["feature"] + 3.0 * terms["localization"]
    assert torch.allclose(terms["total"], expected)


def test_mock_trainer_freezes_teacher_and_backpropagates_student() -> None:
    teacher = torch.nn.Linear(4, 3)
    hook = DistillationTrainerHook(teacher, YOLO26DistillationLoss())
    student_logits = torch.randn(2, 3, requires_grad=True)
    batch = DistillationBatch(student_logits=student_logits, teacher_logits=torch.randn(2, 3))
    loss = MockDistillationTrainer(hook).train_step(student_logits.sum() * 0.0, batch)
    assert loss.requires_grad and student_logits.grad is not None
    assert not teacher.training and all(not parameter.requires_grad for parameter in teacher.parameters())


def test_adapter_dry_run_keeps_student_model_config_unchanged(tmp_path: Path) -> None:
    context = _context(tmp_path)
    preview = YOLO26DistillationAdapter().prepare_patch({"model": "yolo26n.pt"}, {"imgsz": 640}, context, dry_run=True)
    assert preview.patched_model_config == {"model": "yolo26n.pt"}
    assert preview.patched_training_config["distillation"]["teacher"] == "yolo26s.pt"
    assert preview.operations[0].target == "training_config"


def test_teacher_student_protocol_is_enforced() -> None:
    with pytest.raises(ValueError):
        YOLO26DistillationConfig(teacher="yolo26n.pt", student="yolo26n.pt", teacher_data="a", student_data="a")
    with pytest.raises(ValueError):
        YOLO26DistillationConfig(teacher="yolo26s.pt", student="yolo26n.pt", teacher_data="a", student_data="b")
    with pytest.raises(ValueError):
        YOLO26DistillationConfig(teacher="yolo26s.pt", student="yolo26n.pt", teacher_data="a", student_data="a", imgsz=1280)


def test_checkpoint_evidence_records_sha(tmp_path: Path) -> None:
    teacher, student = tmp_path / "teacher.pt", tmp_path / "student.pt"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    evidence = YOLO26DistillationAdapter().build_evidence(teacher, student, _context(tmp_path))
    assert len(evidence.teacher_checkpoint_sha256) == 64 and len(evidence.student_checkpoint_sha256) == 64
    assert evidence.teacher_checkpoint_sha256 != evidence.student_checkpoint_sha256


def test_amp_and_resume_are_preserved_in_patch(tmp_path: Path) -> None:
    preview = YOLO26DistillationAdapter().prepare_patch({}, {}, _context(tmp_path, resume="last.pt", amp=True))
    assert preview.patched_training_config["distillation"]["amp"] is True
    assert preview.patched_training_config["distillation"]["resume"] == "last.pt"


def test_runtime_payload_declares_loss_plugin_amp_resume_and_ddp(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    student = tmp_path / "yolo26n.pt"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    context = _context(
        tmp_path,
        teacher=str(teacher),
        student=str(student),
        teacher_data="coco.yaml",
        student_data="coco.yaml",
    )
    command = [
        "yolo", "detect", "train", f"model={student}", "data=coco.yaml",
        "imgsz=640", f"teacher={teacher}", "feature=true",
    ]

    payload = YOLO26DistillationAdapter().build_runtime_payload(
        context,
        protocol_hash="protocol-1",
        base_command=command,
        generated_config={},
    )

    assert payload.loss_plugin
    assert payload.supports_amp and payload.supports_resume and payload.supports_ddp
    assert {item.name for item in payload.expected_artifacts} == {
        "distillation_evidence"
    }
    payload.verify_imports()
    plugin = YOLO26DistillationRuntimePlugin(**payload.loss_plugin[0].options)
    filtered, _ = plugin.prepare_command(payload=payload, command=command, env={})
    assert not any(item.startswith("teacher=") for item in filtered)
    assert not any(item.startswith("feature=") for item in filtered)


def test_runtime_rejects_dataset_mismatch_and_missing_local_teacher(tmp_path: Path) -> None:
    student_checkpoint = tmp_path / "yolo26n.pt"
    student_checkpoint.write_bytes(b"student")
    plugin = YOLO26DistillationRuntimePlugin(
        teacher=str(tmp_path / "yolo26s.pt"),
        student=str(student_checkpoint),
        teacher_data="coco.yaml",
        student_data="coco.yaml",
        feature_hook_locations=["0"],
    )
    payload = SimpleNamespace()
    with pytest.raises(ValueError, match="runtime data"):
        plugin.prepare_command(
            payload=payload,
            command=[
                "yolo", "detect", "train", f"model={student_checkpoint}",
                "data=other.yaml", "imgsz=640",
            ],
            env={},
        )
    with pytest.raises(FileNotFoundError, match="checkpoint is not a local file"):
        plugin.build_criterion(
            context=_runtime_context(tmp_path),
            trainer=SimpleNamespace(args=SimpleNamespace(imgsz=640, data="coco.yaml")),
            model=torch.nn.Sequential(torch.nn.Linear(2, 2)),
            criterion=object(),
        )


def test_runtime_accepts_only_matching_local_resume_checkpoint(tmp_path: Path) -> None:
    student = tmp_path / "yolo26n.pt"
    resume = tmp_path / "resume_source.pt"
    other = tmp_path / "other.pt"
    for path in (student, resume, other):
        path.write_bytes(path.name.encode("ascii"))
    plugin = YOLO26DistillationRuntimePlugin(
        teacher="yolo26s.pt",
        student=str(student),
        teacher_data="coco.yaml",
        student_data="coco.yaml",
    )
    payload = SimpleNamespace()
    command = [
        "yolo",
        "detect",
        "train",
        f"model={resume}",
        f"resume={resume}",
        "data=coco.yaml",
        "imgsz=640",
    ]

    filtered, _ = plugin.prepare_command(payload=payload, command=command, env={})

    assert f"resume={resume}" in filtered
    with pytest.raises(ValueError, match="runtime model"):
        plugin.prepare_command(
            payload=payload,
            command=[
                "yolo",
                "detect",
                "train",
                f"model={other}",
                f"resume={resume}",
                "data=coco.yaml",
                "imgsz=640",
            ],
            env={},
        )


def test_native_yolo26_loss_plugin_runs_teacher_and_student_backward(tmp_path: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    teacher_checkpoint = tmp_path / "yolo26s.pt"
    student_checkpoint = tmp_path / "yolo26n.pt"
    teacher_checkpoint.write_bytes(b"teacher-checkpoint")
    student_checkpoint.write_bytes(b"student-checkpoint")
    student = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False)
    teacher = DetectionModel("yolo26s.yaml", ch=3, nc=3, verbose=False)
    student.args = get_cfg(overrides={"imgsz": 640})
    teacher.args = get_cfg(overrides={"imgsz": 640})
    student.train()
    plugin = YOLO26DistillationRuntimePlugin(
        teacher=str(teacher_checkpoint),
        student=str(student_checkpoint),
        teacher_data="coco.yaml",
        student_data="coco.yaml",
        imgsz=640,
    )
    plugin._teacher_loader = lambda _: teacher
    context = _runtime_context(tmp_path)
    trainer = SimpleNamespace(
        args=SimpleNamespace(imgsz=640, data="coco.yaml", resume=False)
    )
    criterion = plugin.build_criterion(
        context=context,
        trainer=trainer,
        model=student,
        criterion=student.init_criterion(),
    )
    wrapper = PluginCriterionWrapper(
        criterion,
        _DirectPluginBridge(plugin, context),
        student,
        trainer,
    )
    image = torch.rand(1, 3, 64, 64)
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        predictions = student(image)
        native_loss, _ = criterion(predictions, batch)
        loss, loss_items = wrapper(predictions, batch)
    assert loss.shape == native_loss.shape == loss_items.shape == (3,)
    assert loss.sum() > native_loss.sum()
    loss.sum().backward()

    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    evidence = json.loads(
        (tmp_path / "distillation_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["teacher_forward_calls"] == 1
    assert evidence["teacher_no_grad"] is True
    assert evidence["shared_batch_tensor"] is True
    assert evidence["student_inference_graph_unchanged"] is True
    assert evidence["feature_hook_locations"] == ["model.16", "model.19", "model.22"]
    assert len(evidence["teacher_checkpoint_sha256"]) == 64
    assert len(evidence["student_checkpoint_sha256"]) == 64
    assert all(
        profile["status"] == "method_profile_only"
        and profile["exact_reproduction"] is False
        for profile in evidence["method_profiles"]
    )


def test_distillation_is_disabled_for_validation_loss(tmp_path: Path) -> None:
    plugin = YOLO26DistillationRuntimePlugin(
        teacher="yolo26s.pt",
        student="yolo26n.pt",
        teacher_data="coco.yaml",
        student_data="coco.yaml",
    )
    model = torch.nn.Linear(2, 2).eval()
    native_loss = torch.tensor([1.0, 2.0, 3.0])

    result = plugin.compute_loss(
        context=_runtime_context(tmp_path),
        trainer=object(),
        model=model,
        criterion=object(),
        predictions=object(),
        batch={},
        loss_output=native_loss,
    )

    assert result is native_loss
    assert plugin.teacher is None
    assert plugin._evidence is None


def test_resume_validates_teacher_checkpoint_and_protocol(tmp_path: Path) -> None:
    teacher = tmp_path / "yolo26s.pt"
    student = tmp_path / "yolo26n.pt"
    checkpoint = tmp_path / "last.pt"
    teacher.write_bytes(b"teacher")
    student.write_bytes(b"student")
    checkpoint.write_bytes(b"resume")
    plugin = YOLO26DistillationRuntimePlugin(
        teacher=str(teacher),
        student=str(student),
        teacher_data="coco.yaml",
        student_data="coco.yaml",
    )
    context = _runtime_context(tmp_path)
    state = {
        "config_hash": plugin._config_hash,
        "protocol_hash": "protocol-1",
        "teacher_checkpoint_sha256": _file_sha(teacher),
    }
    sidecar = checkpoint.with_suffix(".pt.distillation.json")
    sidecar.write_text(json.dumps(state), encoding="utf-8")
    trainer = SimpleNamespace(args=SimpleNamespace(resume=str(checkpoint)))

    plugin.on_checkpoint_load(context=context, trainer=trainer, checkpoint={})
    assert plugin._resume_validated is True
    assert plugin._resume_checkpoint == checkpoint.resolve()
    teacher.write_bytes(b"different-teacher")
    with pytest.raises(ValueError, match="teacher_checkpoint_sha256"):
        plugin.on_checkpoint_load(context=context, trainer=trainer, checkpoint={})


def test_checkpoint_serialization_temporarily_removes_feature_hooks(tmp_path: Path) -> None:
    plugin = YOLO26DistillationRuntimePlugin(
        teacher="yolo26s.pt",
        student="yolo26n.pt",
        teacher_data="coco.yaml",
        student_data="coco.yaml",
        feature_hook_locations=["0"],
    )
    student = torch.nn.Sequential(torch.nn.Linear(2, 2))
    teacher = torch.nn.Sequential(torch.nn.Linear(2, 2))
    plugin.student = student
    plugin.teacher = teacher
    plugin._install_feature_hooks(student, teacher)
    ema = copy.deepcopy(student)
    context = _runtime_context(tmp_path)
    trainer = SimpleNamespace(ema=SimpleNamespace(ema=ema))

    assert len(plugin._hook_handles) == 2
    assert len(student[0]._forward_hooks) == 1
    assert len(teacher[0]._forward_hooks) == 1
    assert len(ema[0]._forward_hooks) == 1
    pickle.dumps(ema)

    plugin.on_model_serialize_start(context=context, trainer=trainer)

    assert plugin._hook_handles == []
    assert len(student[0]._forward_hooks) == 0
    assert len(teacher[0]._forward_hooks) == 0
    assert len(ema[0]._forward_hooks) == 0

    plugin.on_model_serialize_end(context=context, trainer=trainer)

    assert len(plugin._hook_handles) == 2
    assert len(student[0]._forward_hooks) == 1
    assert len(teacher[0]._forward_hooks) == 1
    assert len(ema[0]._forward_hooks) == 0


def test_ddp_evidence_is_written_per_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = YOLO26DistillationRuntimePlugin(
        teacher="yolo26s.pt",
        student="yolo26n.pt",
        teacher_data="coco.yaml",
        student_data="coco.yaml",
    )
    plugin._evidence = DistillationEvidence(
        teacher_checkpoint="yolo26s.pt",
        teacher_checkpoint_sha256="a" * 64,
        student_checkpoint="yolo26n.pt",
        student_checkpoint_sha256="b" * 64,
        dataset="coco.yaml",
        split="train",
        rank=2,
    )
    monkeypatch.setenv("RANK", "2")

    plugin._persist_evidence(_runtime_context(tmp_path))

    assert (tmp_path / "distillation_evidence.rank2.json").is_file()
    assert not (tmp_path / "distillation_evidence.json").exists()


def test_component_and_recipe_configs_require_runtime_evidence() -> None:
    contract = load_contracts("configs/components/distillation/yolo26_teacher_student.yaml")[0]
    assert contract.maturity == "adapter_implemented" and not contract.can_execute
    raw = yaml.safe_load(Path("configs/recipes/yolo26n_distillation.yaml").read_text(encoding="utf-8"))
    recipe = recipe_from_mapping(raw)
    assert recipe.train_overrides["imgsz"] == 640 and not recipe.is_executable
    assert recipe.fixed_variables["student_inference_graph"] == "unchanged"


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("YOLO_AGENT_RUN_GPU_TESTS") != "1",
    reason="set YOLO_AGENT_RUN_GPU_TESTS=1 for optional GPU integration test",
)
def test_optional_gpu_runtime_backward(tmp_path: Path) -> None:
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel

    teacher_checkpoint = tmp_path / "yolo26s.pt"
    student_checkpoint = tmp_path / "yolo26n.pt"
    teacher_checkpoint.write_bytes(b"teacher-checkpoint")
    student_checkpoint.write_bytes(b"student-checkpoint")
    student = DetectionModel("yolo26n.yaml", ch=3, nc=3, verbose=False).cuda()
    teacher = DetectionModel("yolo26s.yaml", ch=3, nc=3, verbose=False)
    student.args = get_cfg(overrides={"imgsz": 640})
    teacher.args = get_cfg(overrides={"imgsz": 640})
    student.train()
    plugin = YOLO26DistillationRuntimePlugin(
        teacher=str(teacher_checkpoint),
        student=str(student_checkpoint),
        teacher_data="coco.yaml",
        student_data="coco.yaml",
    )
    plugin._teacher_loader = lambda _: teacher
    context = _runtime_context(tmp_path)
    trainer = SimpleNamespace(
        args=SimpleNamespace(imgsz=640, data="coco.yaml", resume=False)
    )
    criterion = plugin.build_criterion(
        context=context,
        trainer=trainer,
        model=student,
        criterion=student.init_criterion(),
    )
    wrapper = PluginCriterionWrapper(
        criterion,
        _DirectPluginBridge(plugin, context),
        student,
        trainer,
    )
    image = torch.rand(1, 3, 64, 64, device="cuda")
    batch = {
        "img": image,
        "batch_idx": torch.tensor([0], device="cuda"),
        "cls": torch.tensor([[0.0]], device="cuda"),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device="cuda"),
    }
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predictions = student(image)
        loss, _ = wrapper(predictions, batch)
    loss.sum().backward()
    assert any(parameter.grad is not None for parameter in student.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("YOLO_AGENT_RUN_GPU_TESTS") != "1",
    reason="set YOLO_AGENT_RUN_GPU_TESTS=1 for optional GPU adapter smoke",
)
def test_optional_distillation_adapter_gpu_smoke(tmp_path: Path) -> None:
    result = YOLO26DistillationAdapter().gpu_smoke_test(_context(tmp_path))

    assert result.passed, result.errors
    assert result.checks["student_backward"] is True
    assert result.checks["teacher_no_grad"] is True
    assert result.checks["zero_weight_native_equivalent"] is True
    assert result.checks["exact_reproduction_false"] is True


class _DirectPluginBridge:
    def __init__(self, plugin: YOLO26DistillationRuntimePlugin, context: object) -> None:
        self.plugin = plugin
        self.context = context

    def invoke_transform(self, hook: str, value: object, **kwargs: object) -> object:
        assert hook == "compute_loss"
        return self.plugin.compute_loss(
            context=self.context,
            loss_output=value,
            **kwargs,
        )


def _runtime_context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        payload_path=tmp_path / "adapter_runtime_payload.yaml",
        payload=SimpleNamespace(protocol_hash="protocol-1"),
    )


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
