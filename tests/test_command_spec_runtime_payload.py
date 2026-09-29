"""Defense-in-depth tests for runtime payload command wrapping."""

from __future__ import annotations

from yolo_agent.core.command_spec import CommandSpec


def test_runtime_payload_strips_non_cfg_hook_keys_from_stale_argv() -> None:
    """ASHA trials persist registration-time argv; strip stale hook keys.

    Regression: a trial registered before the component-hook whitelist
    re-issued its pre-fix argv every round, re-leaking
    loss.*.weight into the CLI (r13 adapter_runtime_failed) even though
    command_from_training_config was already fixed.
    """
    command = CommandSpec.ultralytics_train(
        model="yolo26n.pt",
        data="coco.yaml",
        project="runs/ultralytics",
        name="legacy_trial_pilot_3",
        batch=32,
        device=0,
    )
    # Persisted ASHA trial commands carry the stale keys in BOTH args and
    # argv; with_runtime_payload prefers self.argv, so the simulation must
    # pollute both to exercise the real r13 shape.
    stale_args = [
        *command.args,
        "fraction=0.1",
        "cos_lr=True",
        "loss.distillation.classifier_response.weight=1.0",
    ]
    stale_argv = [
        *command.argv,
        "fraction=0.1",
        "cos_lr=True",
        "loss.distillation.classifier_response.weight=1.0",
    ]
    stale = command.model_copy(update={"args": stale_args, "argv": stale_argv})

    wrapped = stale.with_runtime_payload(
        "payload/adapter_runtime_payload.yaml",
        runtime_entrypoint="yolo_agent.adapters.ultralytics.runtime_entrypoint",
        payload_hash="hash",
        protocol_hash="protocol",
    )

    joined = " ".join(wrapped.argv or [])
    assert "loss.distillation.classifier_response.weight" not in joined
    assert "cos_lr=True" in joined
    assert "fraction=0.1" in joined
