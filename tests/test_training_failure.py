"""Training failure diagnoser tests."""

from __future__ import annotations

from yolo_agent.agents.training_failure import TrainingFailureDiagnoser, TrainingRunSignals


def test_overfitting_maps_to_reduce_size_or_more_augmentation() -> None:
    """Overfitting should recommend capacity reduction or stronger augmentation."""
    report = TrainingFailureDiagnoser.from_yaml().diagnose(
        TrainingRunSignals(train_loss=0.2, val_loss=0.7)
    )

    modes = {diagnosis.mode for diagnosis in report.diagnoses}
    assert "overfitting" in modes
    overfit = next(diagnosis for diagnosis in report.diagnoses if diagnosis.mode == "overfitting")
    assert "reduce_model_size" in overfit.actions
    assert "increase_augmentation" in overfit.actions


def test_underfitting_maps_to_capacity_or_less_augmentation() -> None:
    """Underfitting should recommend more capacity or reduced augmentation."""
    report = TrainingFailureDiagnoser.from_yaml().diagnose(
        TrainingRunSignals(train_loss=1.4, val_loss=1.5)
    )

    underfit = next(diagnosis for diagnosis in report.diagnoses if diagnosis.mode == "underfitting")
    assert "increase_model_capacity" in underfit.actions
    assert "reduce_augmentation" in underfit.actions


def test_unstable_loss_maps_to_assigner_lr_batch_checks() -> None:
    """Unstable loss should point at assigner, LR, and batch size."""
    report = TrainingFailureDiagnoser.from_yaml().diagnose(
        TrainingRunSignals(loss_history=[1.0, 1.8, 0.9, 1.7])
    )

    unstable = next(diagnosis for diagnosis in report.diagnoses if diagnosis.mode == "unstable_loss")
    assert "check_assigner" in unstable.actions
    assert "reduce_learning_rate" in unstable.actions
    assert "increase_batch_size_or_accumulate" in unstable.actions


def test_low_recall_and_low_precision_actions() -> None:
    """Low recall and low precision should map to distinct action families."""
    report = TrainingFailureDiagnoser.from_yaml().diagnose(
        TrainingRunSignals(recall=0.3, precision=0.4)
    )
    by_mode = {diagnosis.mode: diagnosis for diagnosis in report.diagnoses}

    assert "audit_label_noise" in by_mode["low_recall"].actions
    assert "inspect_small_object_misses" in by_mode["low_recall"].actions
    assert "add_hard_negative_mining" in by_mode["low_precision"].actions
    assert "add_background_only_images" in by_mode["low_precision"].actions


def test_healthy_signals_have_no_failure_diagnosis() -> None:
    """Good signals should not produce failure diagnoses."""
    report = TrainingFailureDiagnoser.from_yaml().diagnose(
        TrainingRunSignals(
            train_loss=0.4,
            val_loss=0.5,
            loss_history=[0.8, 0.7, 0.6],
            recall=0.8,
            precision=0.85,
        )
    )

    assert report.ok is True
    assert report.diagnoses == []

