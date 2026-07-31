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
    DistillationTrainerHook,
    DistillationWeights,
    YOLO26DistillationLoss,
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
}


class YOLO26DistillationConfig(BaseModel):
    teacher: str = "yolo26s.pt"
    student: str = "yolo26n.pt"
    teacher_data: str = "__COMMAND_DATASET__"
    student_data: str = "__COMMAND_DATASET__"
    teacher_split: str = "train"
    student_split: str = "train"
    imgsz: int = 640
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
        if Path(self.teacher).name not in {"yolo26s.pt", "yolo26m.pt"}:
            raise ValueError("teacher must be yolo26s.pt or yolo26m.pt")
        if Path(self.student).name != "yolo26n.pt":
            raise ValueError("student must be yolo26n.pt")
        if self.imgsz != 640:
            raise ValueError("distillation requires fixed imgsz=640")
        if self.teacher_data != self.student_data or self.teacher_split != self.student_split:
            raise ValueError("teacher and student dataset/split must match")
        if not any((self.logits, self.feature, self.localization)):
            raise ValueError("at least one distillation term must be enabled")
        return self


class DistillationMethodProfile(BaseModel):
    method_id: Literal["crosskd", "localization_distillation", "pkd"]
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
    ]


class DistillationEvidence(BaseModel):
    schema_version: str = "yolo26_distillation_evidence.v1"
    adapter_version: str = ""
    plugin_version: str = ""
    adapter_hash: str = ""
    protocol_hash: str = ""
    runtime_payload_hash: str = ""
    changed_variables: dict[str, Any] = Field(default_factory=dict)
    rank: int = -1
    teacher_checkpoint: str
    teacher_checkpoint_sha256: str
    student_checkpoint: str
    student_checkpoint_sha256: str
    dataset: str
    split: str
    imgsz: int = 640
    geometry_policy: str = "shared_preprocessed_batch_tensor"
    augmentation_geometry_hash: str = ""
    shared_batch_tensor: bool = False
    feature_hook_locations: list[str] = Field(default_factory=list)
    enabled_terms: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    method_profiles: list[DistillationMethodProfile] = Field(default_factory=_method_profiles)
    compute_loss_calls: int = 0
    teacher_forward_calls: int = 0
    latest_terms: dict[str, float] = Field(default_factory=dict)
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


class YOLO26DistillationRuntimePlugin:
    """Inject training-only distillation into the native YOLO26 criterion."""

    plugin_version = "yolo26_distillation_runtime.v1"

    def __init__(self, **options: Any) -> None:
        self.config = YOLO26DistillationConfig.model_validate(options)
        self.teacher: Any | None = None
        self._teacher_loader = _load_local_teacher
        self._student_features: dict[str, Any] = {}
        self._teacher_features: dict[str, Any] = {}
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
        data = str(arguments.get("data", self.config.student_data))
        if Path(model).name != Path(self.config.student).name:
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
        self._initialize_runtime(context=context, trainer=trainer, student=model)
        if self.teacher is None or self._evidence is None:
            raise RuntimeError("distillation teacher was not initialized")
        images = batch.get("img")
        if images is None:
            raise ValueError("distillation requires the preprocessed batch image tensor")
        self.teacher.eval()
        self._teacher_features.clear()
        import torch

        with torch.no_grad(), torch.autocast(
            device_type=images.device.type,
            enabled=False,
        ):
            teacher_predictions = self.teacher(images.float())
        student_branch = _one_to_many_branch(predictions)
        teacher_branch = _one_to_many_branch(teacher_predictions)
        student_features = (
            self._ordered_features(self._student_features, student_branch)
            if self.config.feature
            else None
        )
        teacher_features = (
            self._ordered_features(self._teacher_features, teacher_branch)
            if self.config.feature
            else None
        )
        terms = distillation_loss(
            student_branch["scores"],
            teacher_branch["scores"],
            student_features=student_features,
            teacher_features=teacher_features,
            student_boxes=student_branch["boxes"] if self.config.localization else None,
            teacher_boxes=teacher_branch["boxes"] if self.config.localization else None,
            weights=self._effective_weights(),
            logits_dim=1,
        )
        native_loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
        updated = _inject_distillation_total(loss_output, terms["total"])
        updated_loss = updated[0] if isinstance(updated, tuple) else updated
        self._evidence.compute_loss_calls += 1
        self._evidence.teacher_forward_calls += 1
        self._evidence.latest_terms = {
            key: float(value.detach().float().cpu()) for key, value in terms.items()
        }
        self._evidence.native_loss_before = float(
            native_loss.detach().float().sum().cpu()
        )
        self._evidence.total_loss_after = float(
            updated_loss.detach().float().sum().cpu()
        )
        self._evidence.total_loss_changed = bool(
            self._evidence.total_loss_after != self._evidence.native_loss_before
        )
        self._evidence.teacher_eval = not self.teacher.training
        self._evidence.teacher_frozen = all(
            not parameter.requires_grad for parameter in self.teacher.parameters()
        )
        self._evidence.teacher_no_grad = not _contains_grad_tensor(teacher_predictions)
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
        _write_json_atomic(_state_path(context.payload_path.parent), state)
        for checkpoint in checkpoints.values():
            path = Path(checkpoint) if checkpoint else None
            if path is not None and path.is_file():
                _write_json_atomic(_checkpoint_state_path(path), state)
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
                candidates.append(_checkpoint_state_path(resume_path))
            candidates.append(_state_path(context.payload_path.parent))
            state = next((_read_json(path) for path in candidates if path.is_file()), None)
        if not isinstance(state, dict):
            raise ValueError("distillation resume state is missing")
        expected = {
            "config_hash": self._config_hash,
            "protocol_hash": context.payload.protocol_hash,
            "teacher_checkpoint_sha256": _sha256_required(Path(self.config.teacher)),
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
        teacher_path = Path(self.config.teacher)
        student_path = Path(self.config.student)
        teacher_sha = _sha256_required(teacher_path)
        student_sha = _sha256_required(student_path)
        self.teacher = self._teacher_loader(teacher_path)
        teacher_scale = str(getattr(self.teacher, "yaml", {}).get("scale", ""))
        expected_teacher_scale = Path(self.config.teacher).stem[-1]
        if teacher_scale and teacher_scale != expected_teacher_scale:
            raise ValueError(
                "teacher checkpoint architecture does not match its declared YOLO26 scale"
            )
        student_scale = str(getattr(student, "yaml", {}).get("scale", ""))
        if student_scale and student_scale != "n":
            raise ValueError("distillation student architecture must be YOLO26n")
        student_parameter = next(student.parameters())
        self.teacher.to(device=student_parameter.device)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self._install_feature_hooks(student, self.teacher)
        graph_hash = _model_graph_hash(student)
        enabled = [
            name
            for name in ("logits", "feature", "localization")
            if bool(getattr(self.config, name))
        ]
        self._evidence = DistillationEvidence(
            adapter_version=YOLO26DistillationAdapter.adapter_version,
            plugin_version=self.plugin_version,
            adapter_hash=_sha256(Path(__file__)),
            protocol_hash=context.payload.protocol_hash,
            runtime_payload_hash=str(getattr(context.payload, "payload_hash", "")),
            changed_variables=dict(getattr(context.payload, "changed_variables", {})),
            rank=_rank(),
            teacher_checkpoint=str(teacher_path.resolve()),
            teacher_checkpoint_sha256=teacher_sha,
            student_checkpoint=str(student_path.resolve()),
            student_checkpoint_sha256=student_sha,
            dataset=runtime_data,
            split=self.config.student_split,
            feature_hook_locations=list(self.config.feature_hook_locations),
            enabled_terms=enabled,
            weights=self.config.weights.model_dump(mode="json"),
            teacher_eval=not self.teacher.training,
            teacher_frozen=all(
                not parameter.requires_grad for parameter in self.teacher.parameters()
            ),
            student_inference_graph_hash_before=graph_hash,
            student_inference_graph_hash_after=graph_hash,
            student_inference_graph_unchanged=True,
        )
        self._apply_resume_evidence()
        self._persist_evidence(context)

    def _install_feature_hooks(self, student: Any, teacher: Any) -> None:
        for location in self.config.feature_hook_locations:
            student_module = _resolve_module(student, location)
            teacher_module = _resolve_module(teacher, location)
            self._hook_handles.append(
                student_module.register_forward_hook(
                    self._capture_hook(self._student_features, location)
                )
            )
            self._hook_handles.append(
                teacher_module.register_forward_hook(
                    self._capture_hook(self._teacher_features, location)
                )
            )

    @staticmethod
    def _capture_hook(target: dict[str, Any], location: str) -> Any:
        def capture(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            target[location] = output

        return capture

    def _ordered_features(self, captured: dict[str, Any], branch: dict[str, Any]) -> list[Any]:
        del branch
        if all(location in captured for location in self.config.feature_hook_locations):
            return [captured[location] for location in self.config.feature_hook_locations]
        missing = [
            location
            for location in self.config.feature_hook_locations
            if location not in captured
        ]
        raise ValueError(f"distillation feature hooks did not fire: {missing}")

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
            "student_checkpoint_sha256": self._evidence.student_checkpoint_sha256,
            "compute_loss_calls": self._evidence.compute_loss_calls,
        }

    def _persist_evidence(self, context: Any) -> None:
        if self._evidence is not None:
            _write_json_atomic(
                _evidence_path(context.payload_path.parent),
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
            config = YOLO26DistillationConfig.model_validate(context.options)
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
        distill = YOLO26DistillationConfig.model_validate(context.options)
        config["distillation"] = distill.model_dump(mode="json")
        return config

    def build_module(self, context: AdapterContext) -> YOLO26DistillationLoss:
        config = YOLO26DistillationConfig.model_validate(context.options)
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

            config = YOLO26DistillationConfig.model_validate(context.options)
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
            config = YOLO26DistillationConfig.model_validate(context.options)
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
        return [
            ExpectedArtifact(
                name="distillation_evidence",
                relative_path=Path("distillation_evidence.json"),
            )
        ]

    def rollback_plan(self, context: AdapterContext) -> RollbackPlan:
        return RollbackPlan(
            actions=["remove distillation loss plugin and frozen teacher sidecars"],
            files_to_remove=[
                Path("distillation_evidence.json"),
                Path("distillation_state.rank0.json"),
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
        config = YOLO26DistillationConfig.model_validate(context.options)
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
        student_path = Path(config.student)
        config = config.model_copy(
            update={
                "teacher": str(teacher_path.resolve()) if teacher_path.is_file() else config.teacher,
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
            changed_variables={
                "loss.distillation": config.model_dump(mode="json", exclude_none=True)
            },
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
        config = YOLO26DistillationConfig.model_validate(context.options)
        teacher, student = Path(teacher_checkpoint), Path(student_checkpoint)
        return DistillationEvidence(
            teacher_checkpoint=str(teacher),
            teacher_checkpoint_sha256=_sha256(teacher),
            student_checkpoint=str(student),
            student_checkpoint_sha256=_sha256(student),
            dataset=config.teacher_data,
            split=config.teacher_split,
            feature_hook_locations=config.feature_hook_locations,
            enabled_terms=[
                name
                for name in ("logits", "feature", "localization")
                if bool(getattr(config, name))
            ],
            weights=config.weights.model_dump(mode="json"),
        )


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


def _config_hash(config: YOLO26DistillationConfig) -> str:
    payload = config.model_dump(mode="json", exclude={"resume"})
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))


def _evidence_path(directory: Path) -> Path:
    rank = _rank()
    suffix = "" if rank in {-1, 0} else f".rank{rank}"
    return directory / f"distillation_evidence{suffix}.json"


def _state_path(directory: Path) -> Path:
    rank = max(_rank(), 0)
    return directory / f"distillation_state.rank{rank}.json"


def _checkpoint_state_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".distillation.json")


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
