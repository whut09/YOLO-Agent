from pathlib import Path

from yolo_agent.adapters.ultralytics.yolo26_native_audit import YOLO26NativeAuditor
from yolo_agent.adapters.ultralytics.yolo26_native_recipe import YOLO26NativeRecipe
from yolo_agent.adapters.ultralytics.training import UltralyticsTrainingConfig


def test_native_recipe_config_is_fixed_at_640() -> None:
    recipe = YOLO26NativeRecipe.from_yaml()
    assert recipe.imgsz == 640
    assert recipe.fixed_variables["imgsz"] == 640
    assert recipe.fixed_variables["end2end"] is True
    assert recipe.fixed_variables["reg_max"] == 1


def test_audit_reads_installed_ultralytics_and_config_without_training() -> None:
    audit = YOLO26NativeAuditor().audit("configs/training/yolo26_coco_goal.yaml")
    assert audit.ultralytics_version
    assert audit.matched["head_mode"].observed is True
    assert audit.matched["nms_free"].observed is True
    assert audit.matched["dfl_free"].observed == 1
    assert audit.unknown["musgd"].configured == "auto"
    assert "progressive_loss" in audit.unsupported
    assert "stal" in audit.unsupported
    assert audit.recipe_hash and len(audit.recipe_hash) == 64


def test_audit_marks_explicit_non_native_settings_mismatched(tmp_path: Path) -> None:
    config = tmp_path / "train.yaml"
    config.write_text("training:\n  model: yolo26n.pt\n  data: coco.yaml\n  imgsz: 1280\n  optimizer: AdamW\n  batch: 32\n", encoding="utf-8")
    audit = YOLO26NativeAuditor().audit(config)
    assert audit.mismatched["imgsz"].configured == 1280
    assert audit.mismatched["optimizer"].configured == "AdamW"


def test_audit_accepts_typed_training_config() -> None:
    config = UltralyticsTrainingConfig(data=Path("coco.yaml"), model="yolo26n.pt", imgsz=640, batch=48, optimizer="auto")
    audit = YOLO26NativeAuditor().audit(config)
    assert audit.effective_training_config["imgsz"] == 640
    assert audit.effective_training_config["batch"] == 48


def test_runtime_model_audit_checks_end2end() -> None:
    checkpoint = Path("yolo26n.pt")
    audit = YOLO26NativeAuditor().audit("configs/training/yolo26_coco_goal.yaml", model_path=checkpoint if checkpoint.exists() else None)
    assert "runtime_end2end" in audit.matched or "runtime_end2end" in audit.unknown


def test_native_recipe_audit_facade() -> None:
    audit = YOLO26NativeRecipe.from_yaml().audit(config_path="configs/training/yolo26_coco_goal.yaml")
    assert audit.recipe_hash
