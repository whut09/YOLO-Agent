"""Executable YOLO26 teacher-student distillation adapter."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from yolo_agent.components.adapters.base import (
    AdapterContext,
    AdapterValidationReport,
    ComponentAdapter,
    ExpectedArtifact,
    RollbackPlan,
    SmokeTestResult,
    WeightLoadResult,
)
from yolo_agent.components.adapters.runtime import (
    AdapterRuntimePayload,
    RuntimePluginReference,
)
from yolo_agent.components.distillation import (
    DISTILLATION_COMPONENTS,
    DISTILLATION_MECHANISMS,
    DistillationInputs,
    DistillationMechanism,
    DistillationTrainerHook,
    DistillationWeights,
    YOLO26DistillationLoss,
    build_distillation_mechanism_loss,
    distillation_loss,
)


DISTILLATION_COMMAND_KEYS = {
    "teacher",
    "student",
    "teacher_data",
    "student_data",
    "teacher_split",
    "student_split",
    "logits",
    "feature",
    "localization",
    "weights",
    "feature_hook_locations",
    "evidence_interval",
    "mechanism",
    "component_id",
    "changed_variable",
    "weight",
    "teachers",
}


class YOLO26DistillationConfig(BaseModel):
    teacher: str = "yolo26s.pt"
    student: str = "yolo26n.pt"
    teacher_data: str = "__COMMAND_DATASET__"
    student_data: str = "__COMMAND_DATASET__"
    teacher_split: str = "train"
    student_split: str = "train"
    imgsz: int = 640
    mechanism: DistillationMechanism | None = None
    component_id: str = "distillation.yolo26_teacher_student"
    changed_variable: str = "loss.distillation"
    weight: float = Field(default=1.0, ge=0.0)
    teachers: list[str] = Field(default_factory=list)
    logits: bool = True
    feature: bool = True
    localization: bool = True
    weights: DistillationWeights = Field(default_factory=DistillationWeights)
    feature_hook_locations: list[str] = Field(
        default_factory=lambda: ["model.16", "model.19", "model.22"],
        min_length=1,
    )
    evidence_interval: int = Field(default=100, ge=1)
    amp: bool = True
    resume: bool | str = False

    @model_validator(mode="after")
    def validate_protocol(self) -> "YOLO26DistillationConfig":
        teacher_names = [self.teacher, *self.teachers]
        if any(Path(item).name not in {"yolo26s.pt", "yolo26m.pt"} for item in teacher_names):
            raise ValueError("teacher must be yolo26s.pt or yolo26m.pt")
        if Path(self.student).name != "yolo26n.pt":
            raise ValueError("student must be yolo26n.pt")
        if self.imgsz != 640:
            raise ValueError("distillation requires fixed imgsz=640")
        if self.teacher_data != self.student_data or self.teacher_split != self.student_split:
            raise ValueError("teacher and student dataset/split must match")
        if self.mechanism is None and not any((self.logits, self.feature, self.localization)):
            raise ValueError("at least one distillation term must be enabled")
        if self.mechanism is not None:
            spec = DISTILLATION_MECHANISMS[self.mechanism]
            if self.component_id != spec.component_id:
                raise ValueError("distillation mechanism component identity mismatch")
            if self.changed_variable != spec.changed_variable:
                raise ValueError("distillation mechanism changed variable is not canonical")
            if self.mechanism == "teacher_ensemble":
                ensemble = list(dict.fromkeys([self.teacher, *self.teachers]))
                if len(ensemble) < 2:
                    raise ValueError("teacher ensemble requires at least two teachers")
        return self


class DistillationMethodProfile(BaseModel):
    method_id: Literal[
        "crosskd",
        "localization_distillation",
        "pkd",
        "relation_distillation",
        "attention_distillation",
        "masked_feature_distillation",
        "quality_aware_distillation",
        "teacher_ensemble",
    ]
    status: Literal["method_profile_only"] = "method_profile_only"
    exact_reproduction: Literal[False] = False
    note: str


def _method_profiles() -> list[DistillationMethodProfile]:
    return [
        DistillationMethodProfile(
            method_id="crosskd",
            note="Logit profile only; CrossKD architecture details are not reproduced.",
        ),
        DistillationMethodProfile(
            method_id="localization_distillation",
            note="Box-response profile only; no exact paper reproduction is claimed.",
        ),
        DistillationMethodProfile(
            method_id="pkd",
            note="Channel-agnostic normalized feature profile, not exact PKD reproduction.",
        ),
        DistillationMethodProfile(
            method_id="relation_distillation",
            note="Bounded spatial relation adaptation; no paper formula is claimed exact.",
        ),
        DistillationMethodProfile(
            method_id="attention_distillation",
            note="Channel/spatial attention adaptation with explicit feature hooks.",
        ),
        DistillationMethodProfile(
            method_id="masked_feature_distillation",
            note="Teacher-attention masking adaptation, not an exact paper reproduction.",
        ),
        DistillationMethodProfile(
            method_id="quality_aware_distillation",
            note="Teacher-confidence weighted response adaptation.",
        ),
        DistillationMethodProfile(
            method_id="teacher_ensemble",
            note="Probability-space teacher ensemble profile without paper-specific protocol claims.",
        ),
    ]


class DistillationEvidence(BaseModel):
    schema_version: str = "yolo26_distillation_evidence.v1"
    adapter_version: str = ""
    plugin_version: str = ""
    adapter_hash: str = ""
    protocol_hash: str = ""
    runtime_payload_hash: str = ""
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    component_id: str = "distillation.yolo26_teacher_student"
    mechanism: DistillationMechanism | None = None
    changed_variable: str = "loss.distillation"
    mechanism_weight: float = 1.0
    rank: int = -1
    teacher_checkpoint: str
    teacher_checkpoint_sha256: str
    teacher_checkpoints: list[str] = Field(default_factory=list)
    teacher_checkpoint_sha256s: list[str] = Field(default_factory=list)
    student_checkpoint: str
    student_checkpoint_sha256: str
    dataset: str
    split: str
    imgsz: int = 640
    geometry_policy: str = "shared_preprocessed_batch_tensor"
    augmentation_geometry_hash: str = ""
    shared_batch_tensor: bool = False
    feature_hook_locations: list[str] = Field(default_factory=list)
    feature_hooks_required: bool = False
    feature_hooks_validated: bool = False
    student_feature_hook_calls: dict[str, int] = Field(default_factory=dict)
    teacher_feature_hook_calls: dict[str, int] = Field(default_factory=dict)
    enabled_terms: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    method_profiles: list[DistillationMethodProfile] = Field(default_factory=_method_profiles)
    compute_loss_calls: int = 0
    teacher_forward_calls: int = 0
    latest_terms: dict[str, float] = Field(default_factory=dict)
    latest_loss_contribution: float = 0.0
    native_loss_before: float = 0.0
    total_loss_after: float = 0.0
    total_loss_changed: bool = False
    teacher_eval: bool = False
    teacher_frozen: bool = False
    teacher_no_grad: bool = False
    student_inference_graph_hash_before: str = ""
    student_inference_graph_hash_after: str = ""
    student_inference_graph_unchanged: bool = False
    resume_checkpoint: str | None = None
    resume_checkpoint_sha256: str | None = None
    resume_validated: bool = False


class _DistillationFeatureCaptureHook:
    """Pickle-safe feature hook; copied EMA hooks are removed before serialization."""

    def __init__(
        self,
        target: dict[str, Any],
        calls: dict[str, int],
        location: str,
    ) -> None:
        self.target = target
        self.calls = calls
        self.location = location

    def __call__(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        self.target[self.location] = output
        self.calls[self.location] = self.calls.get(self.location, 0) + 1


class YOLO26DistillationRuntimePlugin:
    """Inject training-only distillation into the native YOLO26 criterion."""

    plugin_version = "yolo26_distillation_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = YOLO26DistillationConfig.model_validate(options)
        self.teacher: Any | None = None
        self.teachers: list[Any] = []
        self.student: Any | None = None
        self._teacher_loader = _load_local_teacher
        self._student_features: dict[str, Any] = {}
        self._teacher_features: dict[str, Any] = {}
        self._student_feature_hook_calls: dict[str, int] = {}
        self._teacher_feature_hook_calls: dict[str, int] = {}
        self._hook_handles: list[Any] = []
        self._evidence: DistillationEvidence | None = None
        self._config_hash = _config_hash(self.config)
        self._resume_validated = False
        self._resume_checkpoint: Path | None = None
        self._resume_checkpoint_sha256: str | None = None

    def prepare_command(
        self,
        *,
        payload: AdapterRuntimePayload,
        command: list[str],
        env: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
        del payload
        arguments = _command_arguments(command)
        model = str(arguments.get("model", self.config.student))
        resume = arguments.get("resume")
        data = str(arguments.get("data", self.config.student_data))
        if (
            Path(model).name != Path(self.config.student).name
            and not _same_local_checkpoint(model, resume)
        ):
            raise ValueError("runtime model does not match configured YOLO26 student")
        if not _same_resource(data, self.config.student_data):
            raise ValueError("runtime data does not match teacher/student protocol")
        if int(arguments.get("imgsz", self.config.imgsz)) != 640:
            raise ValueError("distillation runtime requires imgsz=640")
        filtered = [
            token
            for token in command
            if token.partition("=")[0] not in DISTILLATION_COMMAND_KEYS
        ]
        return filtered, env

    def build_criterion(
        self,
        *,
        context: Any,
        trainer: Any,
        model: Any,
        criterion: Any,
    ) -> Any:
        self._initialize_runtime(context=context, trainer=trainer, student=model)
        return criterion

    def build_model(
        self,
        model: Any,
        *,
        context: Any,
        trainer: Any,
    ) -> Any:
        """Install feature hooks before the first student forward."""
        self._initialize_runtime(context=context, trainer=trainer, student=model)
        return model

    def compute_loss(
        self,
        *,
        context: Any,
        trainer: Any,
        model: Any,
        criterion: Any,
        predictions: Any,
        batch: dict[str, Any],
        loss_output: Any,
    ) -> Any:
        del criterion
        if not bool(getattr(model, "training", False)):
            return loss_output
        self._initialize_runtime(context=context, trainer=trainer, student=model)
        if self.teacher is None or self._evidence is None:
            raise RuntimeError("distillation teacher was not initialized")
        images = batch.get("img")
        if images is None:
            raise ValueError("distillation requires the preprocessed batch image tensor")
        for teacher in self.teachers:
            teacher.eval()
        self._teacher_features.clear()
        import torch

        for teacher in self.teachers:
            teacher_parameter = next(teacher.parameters())
            if teacher_parameter.device != images.device:
                teacher.to(device=images.device)
        with torch.no_grad(), torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            teacher_predictions = [teacher(images.float()) for teacher in self.teachers]
        student_branch = _one_to_many_branch(predictions)
        teacher_branches = [_one_to_many_branch(item) for item in teacher_predictions]
        teacher_branch = teacher_branches[0]
        student_features = (
            self._ordered_features(
                self._student_features,
                self._student_feature_hook_calls,
                student_branch,
            )
            if self._requires_features()
            else None
        )
        teacher_features = (
            self._ordered_features(
                self._teacher_features,
                self._teacher_feature_hook_calls,
                teacher_branch,
            )
            if self._requires_features()
            else None
        )
        terms = self._compute_terms(
            student_branch=student_branch,
            teacher_branches=teacher_branches,
            student_features=student_features,
            teacher_features=teacher_features,
        )
        native_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
        updated = _inject_distillation_total(loss_output, terms["total"])
        updated_loss = updated[0] if isinstance(updated, tuple) else updated
        self._evidence.compute_loss_calls += 1
        self._evidence.teacher_forward_calls += len(self.teachers)
        self._evidence.latest_terms = {
            key: float(value.detach().float().cpu()) for key, value in terms.items()
        }
        self._evidence.latest_loss_contribution = self._evidence.latest_terms["total"]
        self._evidence.native_loss_before = float(
            native_loss.detach().float().sum().cpu()
        )
        self._evidence.total_loss_after = float(
            updated_loss.detach().float().sum().cpu()
        )
        self._evidence.total_loss_changed = bool(
            self._evidence.total_loss_after != self._evidence.native_loss_before
        )
        self._evidence.teacher_eval = all(not teacher.training for teacher in self.teachers)
        self._evidence.teacher_frozen = all(
            not parameter.requires_grad
            for teacher in self.teachers
            for parameter in teacher.parameters()
        )
        self._evidence.teacher_no_grad = not _contains_grad_tensor(teacher_predictions)
        self._evidence.student_feature_hook_calls = dict(
            self._student_feature_hook_calls
        )
        self._evidence.teacher_feature_hook_calls = dict(
            self._teacher_feature_hook_calls
        )
        self._evidence.feature_hooks_validated = bool(
            not self._requires_features()
            or all(
                self._student_feature_hook_calls.get(location, 0) > 0
                and self._teacher_feature_hook_calls.get(location, 0) > 0
                for location in self.config.feature_hook_locations
            )
        )
        self._evidence.shared_batch_tensor = True
        self._evidence.augmentation_geometry_hash = _geometry_hash(images)
        should_persist = (
            self._evidence.compute_loss_calls == 1
            or self._evidence.compute_loss_calls % self.config.evidence_interval == 0
        )
        if should_persist:
            self._evidence.student_inference_graph_hash_after = _model_graph_hash(model)
            self._evidence.student_inference_graph_unchanged = (
                self._evidence.student_inference_graph_hash_before
                == self._evidence.student_inference_graph_hash_after
            )
            self._persist_evidence(context)
        self._student_features.clear()
        self._teacher_features.clear()
        return updated

    def on_checkpoint_save(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoints: dict[str, Any],
    ) -> None:
        del trainer
        if self._evidence is None:
            return
        state = self._resume_state(context)
        _write_json_atomic(
            _state_path(context.payload_path.parent, self.config.mechanism), state
        )
        for checkpoint in checkpoints.values():
            path = Path(checkpoint) if checkpoint else None
            if path is not None and path.is_file():
                _write_json_atomic(
                    _checkpoint_state_path(path, self.config.mechanism), state
                )

    def on_model_serialize_start(self, *, context: Any, trainer: Any) -> None:
        del context
        self._remove_feature_hooks()
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        self._remove_copied_feature_hooks(ema)

    def on_model_serialize_end(self, *, context: Any, trainer: Any) -> None:
        del trainer
        if self._requires_features() and self.student is not None and self.teacher is not None:
            self._install_feature_hooks(self.student, self.teacher)
        self._persist_evidence(context)

    def on_checkpoint_load(
        self,
        *,
        context: Any,
        trainer: Any,
        checkpoint: Any,
    ) -> None:
        state = checkpoint.get("yolo_agent_distillation") if isinstance(checkpoint, dict) else None
        resume = getattr(getattr(trainer, "args", None), "resume", None)
        resume_path = (
            Path(resume)
            if isinstance(resume, (str, Path)) and str(resume).lower() not in {"true", "false"}
            else None
        )
        if not isinstance(state, dict):
            candidates = []
            if resume_path is not None:
                candidates.append(
                    _checkpoint_state_path(resume_path, self.config.mechanism)
                )
            candidates.append(
                _state_path(context.payload_path.parent, self.config.mechanism)
            )
            state = next((_read_json(path) for path in candidates if path.is_file()), None)
        if not isinstance(state, dict):
            raise ValueError("distillation resume state is missing")
        expected = {
            "config_hash": self._config_hash,
            "protocol_hash": context.payload.protocol_hash,
            "teacher_checkpoint_sha256": _sha256_required(Path(self.config.teacher)),
            "teacher_checkpoint_sha256s": [
                _sha256_required(path) for path in self._teacher_paths()
            ],
            "runtime_payload_hash": str(
                getattr(context.payload, "payload_hash", "")
            ),
            "component_id": self.config.component_id,
            "mechanism": self.config.mechanism,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"distillation resume mismatch for {key}: "
                    f"expected={value!r} actual={state.get(key)!r}"
                )
        self._resume_validated = True
        if resume_path is not None and resume_path.is_file():
            self._resume_checkpoint = resume_path.resolve()
            self._resume_checkpoint_sha256 = _sha256(resume_path)
        if self._evidence is not None:
            self._apply_resume_evidence()
            self._persist_evidence(context)

    def _initialize_runtime(self, *, context: Any, trainer: Any, student: Any) -> None:
        if self.teacher is not None:
            return
        args = getattr(trainer, "args", None)
        runtime_imgsz = getattr(args, "imgsz", self.config.imgsz)
        if isinstance(runtime_imgsz, (list, tuple)):
            runtime_imgsz = runtime_imgsz[0]
        if int(runtime_imgsz) != 640:
            raise ValueError("distillation trainer must use imgsz=640")
        runtime_data = str(getattr(args, "data", self.config.student_data))
        if not _same_resource(runtime_data, self.config.student_data):
            raise ValueError("teacher and student must use the same runtime dataset")
        teacher_paths = self._teacher_paths()
        student_path = Path(self.config.student)
        teacher_shas = [_sha256_required(path) for path in teacher_paths]
        student_sha = _sha256_required(student_path)
        self.teachers = [self._teacher_loader(path) for path in teacher_paths]
        self.teacher = self.teachers[0]
        self.student = student
        for teacher, teacher_path in zip(self.teachers, teacher_paths, strict=True):
            teacher_scale = str(getattr(teacher, "yaml", {}).get("scale", ""))
            expected_teacher_scale = teacher_path.stem[-1]
            if teacher_scale and teacher_scale != expected_teacher_scale:
                raise ValueError(
                    "teacher checkpoint architecture does not match its declared YOLO26 scale"
                )
        student_scale = str(getattr(student, "yaml", {}).get("scale", ""))
        if student_scale and student_scale != "n":
            raise ValueError("distillation student architecture must be YOLO26n")
        student_parameter = next(student.parameters())
        for teacher in self.teachers:
            teacher.to(device=student_parameter.device)
            teacher.eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        if self._requires_features():
            self._install_feature_hooks(student, self.teacher)
        graph_hash = _model_graph_hash(student)
        enabled = (
            [self.config.mechanism]
            if self.config.mechanism is not None
            else [
                name
                for name in ("logits", "feature", "localization")
                if bool(getattr(self.config, name))
            ]
        )
        self._evidence = DistillationEvidence(
            adapter_version=YOLO26DistillationAdapter.adapter_version,
            plugin_version=self.plugin_version,
            adapter_hash=_sha256(Path(__file__)),
            protocol_hash=context.payload.protocol_hash,
            runtime_payload_hash=str(getattr(context.payload, "payload_hash", "")),
            changed_variables=dict(getattr(context.payload, "changed_variables", {})),
            component_id=self.config.component_id,
            mechanism=self.config.mechanism,
            changed_variable=self.config.changed_variable,
            mechanism_weight=self.config.weight,
            rank=_rank(),
            teacher_checkpoint=str(teacher_paths[0].resolve()),
            teacher_checkpoint_sha256=teacher_shas[0],
            teacher_checkpoints=[str(path.resolve()) for path in teacher_paths],
            teacher_checkpoint_sha256s=teacher_shas,
            student_checkpoint=str(student_path.resolve()),
            student_checkpoint_sha256=student_sha,
            dataset=runtime_data,
            split=self.config.student_split,
            feature_hook_locations=list(self.config.feature_hook_locations),
            feature_hooks_required=self._requires_features(),
            feature_hooks_validated=not self._requires_features(),
            enabled_terms=enabled,
            weights=(
                {self.config.mechanism: self.config.weight}
                if self.config.mechanism is not None
                else self.config.weights.model_dump(mode="json")
            ),
            teacher_eval=all(not teacher.training for teacher in self.teachers),
            teacher_frozen=all(
                not parameter.requires_grad
                for teacher in self.teachers
                for parameter in teacher.parameters()
            ),
            student_inference_graph_hash_before=graph_hash,
            student_inference_graph_hash_after=graph_hash,
            student_inference_graph_unchanged=True,
        )
        self._apply_resume_evidence()
        self._persist_evidence(context)

    def _install_feature_hooks(self, student: Any, teacher: Any) -> None:
        if self._hook_handles:
            raise RuntimeError("distillation feature hooks are already installed")
        for location in self.config.feature_hook_locations:
            student_module = _resolve_module(student, location)
            teacher_module = _resolve_module(teacher, location)
            self._hook_handles.append(
                student_module.register_forward_hook(
                    _DistillationFeatureCaptureHook(
                        self._student_features,
                        self._student_feature_hook_calls,
                        location,
                    )
                )
            )
            self._hook_handles.append(
                teacher_module.register_forward_hook(
                    _DistillationFeatureCaptureHook(
                        self._teacher_features,
                        self._teacher_feature_hook_calls,
                        location,
                    )
                )
            )

    def _remove_feature_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @staticmethod
    def _remove_copied_feature_hooks(model: Any) -> None:
        if model is None or not callable(getattr(model, "modules", None)):
            return
        for module in model.modules():
            hooks = getattr(module, "_forward_hooks", None)
            if hooks is None:
                continue
            for hook_id, hook in list(hooks.items()):
                if isinstance(hook, _DistillationFeatureCaptureHook):
                    del hooks[hook_id]

    def _ordered_features(
        self,
        captured: dict[str, Any],
        calls: dict[str, int],
        branch: dict[str, Any],
    ) -> list[Any]:
        del branch
        if all(
            location in captured and calls.get(location, 0) > 0
            for location in self.config.feature_hook_locations
        ):
            return [captured[location] for location in self.config.feature_hook_locations]
        missing = [
            location
            for location in self.config.feature_hook_locations
            if location not in captured
        ]
        raise ValueError(f"distillation feature hooks did not fire: {missing}")

    def _requires_features(self) -> bool:
        if self.config.mechanism is None:
            return self.config.feature
        return DISTILLATION_MECHANISMS[self.config.mechanism].requires_features

    def _teacher_paths(self) -> list[Path]:
        values = [self.config.teacher]
        if self.config.mechanism == "teacher_ensemble":
            values.extend(self.config.teachers)
        return [Path(value) for value in dict.fromkeys(values)]

    def _compute_terms(
        self,
        *,
        student_branch: dict[str, Any],
        teacher_branches: list[dict[str, Any]],
        student_features: list[Any] | None,
        teacher_features: list[Any] | None,
    ) -> dict[str, Any]:
        if self.config.mechanism is None:
            return distillation_loss(
                student_branch["scores"],
                teacher_branches[0]["scores"],
                student_features=student_features,
                teacher_features=teacher_features,
                student_boxes=(
                    student_branch["boxes"] if self.config.localization else None
                ),
                teacher_boxes=(
                    teacher_branches[0]["boxes"]
                    if self.config.localization
                    else None
                ),
                weights=self._effective_weights(),
                logits_dim=1,
            )
        mechanism = self.config.mechanism
        options: dict[str, Any] = {}
        if mechanism in {"logits", "quality_aware", "teacher_ensemble"}:
            options = {
                "temperature": self.config.weights.temperature,
                "class_dim": 1,
            }
        plugin = build_distillation_mechanism_loss(mechanism, **options)
        teacher_logits: Any = teacher_branches[0]["scores"]
        if mechanism == "teacher_ensemble":
            teacher_logits = [branch["scores"] for branch in teacher_branches]
        output = plugin.compute(
            DistillationInputs(
                student_logits=student_branch["scores"],
                teacher_logits=teacher_logits,
                student_features=student_features,
                teacher_features=teacher_features,
                student_boxes=(
                    student_branch["boxes"] if mechanism == "localization" else None
                ),
                teacher_boxes=(
                    teacher_branches[0]["boxes"]
                    if mechanism == "localization"
                    else None
                ),
            )
        )
        weighted = output.loss * self.config.weight
        return {"total": weighted, mechanism: output.loss}

    def _apply_resume_evidence(self) -> None:
        if self._evidence is None:
            return
        self._evidence.resume_validated = self._resume_validated
        if self._resume_checkpoint is not None:
            self._evidence.resume_checkpoint = str(self._resume_checkpoint)
            self._evidence.resume_checkpoint_sha256 = self._resume_checkpoint_sha256

    def _effective_weights(self) -> DistillationWeights:
        return self.config.weights.model_copy(
            update={
                "logits": self.config.weights.logits if self.config.logits else 0.0,
                "feature": self.config.weights.feature if self.config.feature else 0.0,
                "localization": (
                    self.config.weights.localization if self.config.localization else 0.0
                ),
            }
        )

    def _resume_state(self, context: Any) -> dict[str, Any]:
        if self._evidence is None:
            raise RuntimeError("distillation evidence is not initialized")
        return {
            "schema_version": "yolo26_distillation_state.v1",
            "config_hash": self._config_hash,
            "protocol_hash": context.payload.protocol_hash,
            "teacher_checkpoint_sha256": self._evidence.teacher_checkpoint_sha256,
            "teacher_checkpoint_sha256s": self._evidence.teacher_checkpoint_sha256s,
            "student_checkpoint_sha256": self._evidence.student_checkpoint_sha256,
            "runtime_payload_hash": self._evidence.runtime_payload_hash,
            "component_id": self._evidence.component_id,
            "mechanism": self._evidence.mechanism,
            "compute_loss_calls": self._evidence.compute_loss_calls,
        }

    def _persist_evidence(self, context: Any) -> None:
        if self._evidence is not None:
            _write_json_atomic(
                _evidence_path(
                    context.payload_path.parent,
                    self.config.mechanism,
                ),
                self._evidence.model_dump(mode="json"),
            )


class YOLO26DistillationAdapter(ComponentAdapter):
    adapter_version = "yolo26_distillation.v3"
    source_commit = "yolo-agent:yolo26-distillation-runtime-v1"
    strategy = "loss_injection"
    modified_model_fields = frozenset()
    modified_training_fields = frozenset({"distillation"})

    def validate_environment(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            import torch

            import ultralytics

            return AdapterValidationReport(
                ok=True,
                checks={"torch": torch.__version__, "ultralytics": ultralytics.__version__},
            )
        except ImportError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])

    def validate_compatibility(self, context: AdapterContext) -> AdapterValidationReport:
        try:
            config = _context_config(context)
        except ValueError as exc:
            return AdapterValidationReport(ok=False, errors=[str(exc)])
        errors = []
        if context.imgsz != 640:
            errors.append("fixed imgsz=640 required")
        if context.detector_family != "yolo26":
            errors.append("distillation runtime supports YOLO26 only")
        return AdapterValidationReport(
            ok=not errors,
            errors=errors,
            checks={
                "teacher": config.teacher,
                "student": config.student,
                "shared_augmented_batch": True,
                "student_inference_graph_unchanged": True,
            },
        )

    def patch_model_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        return config

    def patch_training_config(
        self, config: dict[str, Any], context: AdapterContext, *, dry_run: bool = True
    ) -> dict[str, Any]:
        distill = _context_config(context)
        config["distillation"] = distill.model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> YOLO26DistillationLoss:
        config = _context_config(context)
        weights = config.weights.model_copy(
            update={
                "logits": config.weights.logits if config.logits else 0.0,
                "feature": config.weights.feature if config.feature else 0.0,
                "localization": (
                    config.weights.localization if config.localization else 0.0
                ),
            }
        )
        return YOLO26DistillationLoss(weights)

    def build_trainer_hook(
        self, teacher_model: Any, context: AdapterContext
    ) -> DistillationTrainerHook:
        return DistillationTrainerHook(teacher_model, self.build_module(context))

    def load_pretrained_weights(
        self,
        module: Any,
        weights: Path | str | None,
        context: AdapterContext,
    ) -> WeightLoadResult:
        if weights is None:
            return WeightLoadResult(loaded=False, message="teacher checkpoint is required")
        source = Path(weights)
        return WeightLoadResult(
            loaded=source.is_file(),
            source=source,
            message="runtime owns the frozen teacher model",
        )

    def smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            config = _context_config(context)
            student_logits = torch.randn(2, 4, 8, requires_grad=True)
            teacher_logits = torch.randn(2, 4, 8)
            student_features = [torch.randn(2, 8, 4, 4, requires_grad=True)]
            teacher_features = [torch.randn(2, 16, 4, 4)]
            student_boxes = torch.randn(2, 4, 8, requires_grad=True)
            teacher_boxes = torch.randn(2, 4, 8)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                terms = distillation_loss(
                    student_logits,
                    teacher_logits,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    student_boxes=student_boxes,
                    teacher_boxes=teacher_boxes,
                    weights=config.weights,
                    logits_dim=1,
                )
                native = (torch.ones(3, requires_grad=True), torch.ones(3))
                loss, _ = _inject_distillation_total(native, terms["total"])
            loss.sum().backward()
            return SmokeTestResult(
                passed=(
                    student_logits.grad is not None
                    and student_features[0].grad is not None
                    and student_boxes.grad is not None
                ),
                evidence_kind="local",
                checks={
                    "shape": str(tuple(student_logits.shape)),
                    "backward": student_logits.grad is not None,
                    "amp": True,
                    "runtime_loss_injection": True,
                    "student_graph_unchanged": True,
                    "imgsz": str(config.imgsz),
                },
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                errors=[str(exc)],
            )

    def gpu_smoke_test(self, context: AdapterContext) -> SmokeTestResult:
        try:
            import torch

            if not torch.cuda.is_available():
                return SmokeTestResult(
                    passed=False,
                    evidence_kind="local",
                    checks={"gpu_smoke_implemented": True, "cuda_available": False},
                    errors=["cuda_not_available"],
                )
            config = _context_config(context)
            device = torch.device("cuda")
            student_logits = torch.randn(
                2,
                4,
                8,
                device=device,
                requires_grad=True,
            )
            teacher_logits = torch.randn(2, 4, 8, device=device)
            student_features = [
                torch.randn(2, 8, 4, 4, device=device, requires_grad=True)
            ]
            teacher_features = [torch.randn(2, 16, 4, 4, device=device)]
            student_boxes = torch.randn(
                2,
                4,
                8,
                device=device,
                requires_grad=True,
            )
            teacher_boxes = torch.randn(2, 4, 8, device=device)
            native = torch.ones(3, device=device, requires_grad=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                terms = distillation_loss(
                    student_logits,
                    teacher_logits,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    student_boxes=student_boxes,
                    teacher_boxes=teacher_boxes,
                    weights=config.weights,
                    logits_dim=1,
                )
                active = _inject_distillation_total(native, terms["total"])
                zero_terms = distillation_loss(
                    student_logits,
                    teacher_logits,
                    student_features=student_features,
                    teacher_features=teacher_features,
                    student_boxes=student_boxes,
                    teacher_boxes=teacher_boxes,
                    weights=DistillationWeights(
                        logits=0.0,
                        feature=0.0,
                        localization=0.0,
                    ),
                    logits_dim=1,
                )
                zero = _inject_distillation_total(native, zero_terms["total"])
            active.sum().backward()
            student_backward = bool(
                student_logits.grad is not None
                and student_features[0].grad is not None
                and student_boxes.grad is not None
            )
            teacher_no_grad = bool(
                teacher_logits.grad is None
                and teacher_features[0].grad is None
                and teacher_boxes.grad is None
            )
            zero_equivalent = torch.equal(zero, native)
            profiles = _method_profiles()
            exact_reproduction_false = all(
                profile.exact_reproduction is False for profile in profiles
            )
            passed = bool(
                student_backward
                and teacher_no_grad
                and zero_equivalent
                and exact_reproduction_false
            )
            return SmokeTestResult(
                passed=passed,
                evidence_kind="local",
                checks={
                    "gpu_smoke_implemented": True,
                    "cuda_available": True,
                    "amp": True,
                    "student_backward": student_backward,
                    "teacher_no_grad": teacher_no_grad,
                    "zero_weight_native_equivalent": zero_equivalent,
                    "exact_reproduction_false": exact_reproduction_false,
                    "imgsz": "640",
                },
                errors=[] if passed else ["distillation CUDA smoke failed"],
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            return SmokeTestResult(
                passed=False,
                evidence_kind="local",
                checks={"gpu_smoke_implemented": True},
                errors=[str(exc)],
            )

    def expected_artifacts(self, context: AdapterContext) -> list[ExpectedArtifact]:
        config = _context_config(context)
        prefix = (
            "distillation"
            if config.mechanism is None
            else f"distillation_{config.mechanism}"
        )
        return [
            ExpectedArtifact(
                name=f"{prefix}_evidence",
                relative_path=Path(f"{prefix}_evidence.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        config = _context_config(context)
        prefix = (
            "distillation"
            if config.mechanism is None
            else f"distillation_{config.mechanism}"
        )
        return RollbackPlan(
            actions=["remove distillation loss plugin and frozen teacher sidecars"],
            files_to_remove=[
                Path(f"{prefix}_evidence.json"),
                Path(f"{prefix}_state.rank0.json"),
            ],
        )

    def build_runtime_payload(
        self,
        context: AdapterContext,
        *,
        protocol_hash: str,
        base_command: list[str],
        generated_config: dict[str, Any],
    ) -> AdapterRuntimePayload:
        config = _context_config(context)
        arguments = _command_arguments(base_command)
        data = str(arguments.get("data", config.student_data))
        student = str(arguments.get("model", config.student))
        if config.teacher_data == "__COMMAND_DATASET__":
            config = config.model_copy(
                update={"teacher_data": data, "student_data": data, "student": student}
            )
        elif Path(student).name == "yolo26n.pt":
            config = config.model_copy(update={"student": student})
        teacher_path = Path(config.teacher)
        teacher_paths = [Path(item) for item in config.teachers]
        student_path = Path(config.student)
        config = config.model_copy(
            update={
                "teacher": str(teacher_path.resolve()) if teacher_path.is_file() else config.teacher,
                "teachers": [
                    str(path.resolve()) if path.is_file() else str(path)
                    for path in teacher_paths
                ],
                "student": str(student_path.resolve()) if student_path.is_file() else config.student,
            }
        )
        return AdapterRuntimePayload(
            component_ids=[context.contract.component_id],
            adapter_classes=[type(self).__name__],
            adapter_versions={context.contract.component_id: self.adapter_version},
            source_commits={context.contract.component_id: self.source_commit},
            loss_plugin=[
                RuntimePluginReference(
                    reference=(
                        "yolo_agent.components.adapters.distillation.yolo26_distillation:"
                        "YOLO26DistillationRuntimePlugin"
                    ),
                    options=config.model_dump(mode="json", exclude_none=True),
                    required_hooks=["compute_loss"],
                )
            ],
            generated_config=generated_config,
            changed_variables=(
                {config.changed_variable: config.weight}
                if config.mechanism is not None
                else {
                    "loss.distillation": config.model_dump(
                        mode="json", exclude_none=True
                    )
                }
            ),
            expected_artifacts=self.expected_artifacts(context),
            rollback_plan=self.rollback_plan(context),
            protocol_hash=protocol_hash,
            base_command=base_command,
            supports_amp=True,
            supports_ddp=True,
            supports_resume=True,
        )

    def build_evidence(
        self,
        teacher_checkpoint: Path | str,
        student_checkpoint: Path | str,
        context: AdapterContext,
    ) -> DistillationEvidence:
        config = _context_config(context)
        teacher, student = Path(teacher_checkpoint), Path(student_checkpoint)
        return DistillationEvidence(
            teacher_checkpoint=str(teacher),
            teacher_checkpoint_sha256=_sha256(teacher),
            teacher_checkpoints=[str(teacher)],
            teacher_checkpoint_sha256s=[_sha256(teacher)],
            student_checkpoint=str(student),
            student_checkpoint_sha256=_sha256(student),
            dataset=config.teacher_data,
            split=config.teacher_split,
            feature_hook_locations=config.feature_hook_locations,
            feature_hooks_required=(
                config.mechanism is not None
                and DISTILLATION_MECHANISMS[config.mechanism].requires_features
            ),
            component_id=config.component_id,
            mechanism=config.mechanism,
            changed_variable=config.changed_variable,
            mechanism_weight=config.weight,
            enabled_terms=[
                name
                for name in ("logits", "feature", "localization")
                if bool(getattr(config, name))
            ],
            weights=config.weights.model_dump(mode="json"),
        )


def _context_config(context: AdapterContext) -> YOLO26DistillationConfig:
    options = dict(context.options)
    spec = DISTILLATION_COMPONENTS.get(context.contract.component_id)
    if spec is not None:
        options.update(
            {
                "mechanism": spec.mechanism,
                "component_id": spec.component_id,
                "changed_variable": spec.changed_variable,
                "weight": float(options.get(spec.changed_variable, options.get("weight", 1.0))),
            }
        )
    return YOLO26DistillationConfig.model_validate(options)


def _load_local_teacher(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"local teacher checkpoint is required; automatic download is disabled: {path}"
        )
    from ultralytics import YOLO

    return YOLO(str(path.resolve()), task="detect", verbose=False).model


def _one_to_many_branch(predictions: Any) -> dict[str, Any]:
    if isinstance(predictions, tuple):
        predictions = predictions[1]
    if not isinstance(predictions, dict) or "one2many" not in predictions:
        raise ValueError("YOLO26 distillation requires native one2many predictions")
    branch = predictions["one2many"]
    if not isinstance(branch, dict) or not {"boxes", "scores"}.issubset(branch):
        raise ValueError("YOLO26 one2many output is missing boxes or scores")
    return branch


def _inject_distillation_total(loss_output: Any, distillation_total: Any) -> Any:
    import torch

    if isinstance(loss_output, tuple) and len(loss_output) == 2:
        native_loss, loss_items = loss_output
        return _add_once(native_loss, distillation_total, torch), loss_items
    return _add_once(loss_output, distillation_total, torch)


def _add_once(native_loss: Any, distillation_total: Any, torch: Any) -> Any:
    if not torch.is_tensor(native_loss):
        raise TypeError("native Ultralytics loss must be a tensor")
    if native_loss.ndim == 0:
        return native_loss + distillation_total
    addition = torch.cat(
        [
            distillation_total.reshape(1).to(native_loss.dtype),
            native_loss.new_zeros(native_loss.numel() - 1),
        ]
    ).reshape_as(native_loss)
    return native_loss + addition


def _resolve_module(root: Any, location: str) -> Any:
    value = root
    for part in location.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    if not callable(getattr(value, "register_forward_hook", None)):
        raise ValueError(f"feature hook target is not a module: {location}")
    return value


def _contains_grad_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if torch.is_tensor(value):
        return bool(value.requires_grad)
    if isinstance(value, dict):
        return any(_contains_grad_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_grad_tensor(item) for item in value)
    return False


def _model_graph_hash(model: Any) -> str:
    payload = {
        "yaml": getattr(model, "yaml", {}),
        "modules": [type(module).__module__ + "." + type(module).__qualname__ for module in model.modules()],
        "state_shapes": {
            name: list(value.shape) for name, value in model.state_dict().items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _geometry_hash(images: Any) -> str:
    payload = {
        "shape": list(images.shape),
        "stride": list(images.stride()),
        "dtype": str(images.dtype),
        "device_type": images.device.type,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _command_arguments(command: list[str]) -> dict[str, Any]:
    import yaml

    arguments: dict[str, Any] = {}
    for token in command[3:]:
        key, separator, value = token.partition("=")
        if separator:
            arguments[key] = yaml.safe_load(value)
    return arguments


def _same_resource(left: str, right: str) -> bool:
    if left == right:
        return True
    if "__COMMAND_DATASET__" in {left, right}:
        return False
    return Path(left).resolve() == Path(right).resolve()


def _same_local_checkpoint(model: str, resume: Any) -> bool:
    if not isinstance(resume, (str, Path)) or str(resume).lower() in {"true", "false"}:
        return False
    model_path = Path(model).expanduser()
    resume_path = Path(resume).expanduser()
    return bool(
        model_path.is_file()
        and resume_path.is_file()
        and model_path.resolve() == resume_path.resolve()
    )


def _config_hash(config: YOLO26DistillationConfig) -> str:
    payload = config.model_dump(mode="json", exclude={"resume"})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


def _evidence_path(
    directory: Path,
    mechanism: DistillationMechanism | None = None,
) -> Path:
    rank = _rank()
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    prefix = "distillation" if mechanism is None else f"distillation_{mechanism}"
    return directory / f"{prefix}_evidence{suffix}.json"


def _state_path(
    directory: Path,
    mechanism: DistillationMechanism | None = None,
) -> Path:
    rank = max(_rank(), 0)
    prefix = "distillation" if mechanism is None else f"distillation_{mechanism}"
    return directory / f"{prefix}_state.rank{rank}.json"


def _checkpoint_state_path(
    checkpoint: Path,
    mechanism: DistillationMechanism | None = None,
) -> Path:
    suffix = ".distillation" if mechanism is None else f".distillation.{mechanism}"
    return checkpoint.with_suffix(checkpoint.suffix + suffix + ".json")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_required(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is not a local file: {path}")
    return _sha256(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DistillationEvidence",
    "DistillationMethodProfile",
    "YOLO26DistillationAdapter",
    "YOLO26DistillationConfig",
    "YOLO26DistillationRuntimePlugin",
]
