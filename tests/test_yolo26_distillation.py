from pathlib import Path

import pytest
import torch

from yolo_agent.components.adapters.base import AdapterContext
from yolo_agent.components.adapters.distillation.yolo26_distillation import YOLO26DistillationAdapter, YOLO26DistillationConfig
from yolo_agent.components.contracts import ComponentContract, load_contracts
from yolo_agent.components.distillation import DistillationBatch, DistillationTrainerHook, MockDistillationTrainer, YOLO26DistillationLoss, distillation_loss
from yolo_agent.recipes.schemas import recipe_from_mapping
import yaml


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


def test_component_and_recipe_configs_are_smoke_executable() -> None:
    contract = load_contracts("configs/components/distillation/yolo26_teacher_student.yaml")[0]
    assert contract.maturity == "smoke_passed" and contract.can_execute
    raw = yaml.safe_load(Path("configs/recipes/yolo26n_distillation.yaml").read_text(encoding="utf-8"))
    recipe = recipe_from_mapping(raw)
    assert recipe.train_overrides["imgsz"] == 640 and recipe.is_executable
    assert recipe.fixed_variables["student_inference_graph"] == "unchanged"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optional GPU integration test")
def test_optional_gpu_backward() -> None:
    student = torch.randn(2, 8, device="cuda", requires_grad=True)
    teacher = torch.randn(2, 8, device="cuda")
    distillation_loss(student, teacher)["total"].backward()
    assert student.grad is not None
