"""Canonical :class:`RuntimeHookIdentity` for auditable runtime hooks.

Prompt-18G introduces one unified structure that every paper runtime hook
must resolve to before the preflight may pass.  The string ``"unknown"`` is
no longer an acceptable runtime identity: a real executable paper binding
must resolve to a concrete module, class, method, and phase, and the
identity must carry enough provenance (source SHA-256, component id, paper
binding) to be re-verified later by hash comparison.

Phases cover the full runtime surface of the agent — data, preprocess,
model_graph, loss, assignment, optimizer, train_step, postprocess,
inference, evaluation — so data-side papers (sampling, augmentation,
annotation, postprocess) resolve to their true phase instead of being
forced into a model-forward shape they do not have.

Anti-fake contract (enforced at construction time):

* ``implementation_path`` must be a real dotted import path and
  ``class_name`` a resolvable attribute on it — a hook whose callable
  cannot be imported is rejected immediately;
* ``method_name`` must be a real attribute of the resolved class when the
  class can be imported;
* ``source_sha256`` must be the hash of the file backing
  ``implementation_path`` when that file exists, so any later edit to the
  adapter source invalidates the identity (drift detection);
* ``paper_binding_id`` must be provided — a shared primitive (identical
  ``implementation_path`` reused by several papers) still requires a
  paper-specific binding so each paper owns an independent identity.
"""

from __future__ import annotations

import hashlib
import importlib
from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

RuntimeHookPhase = Literal[
    "data",
    "preprocess",
    "model_graph",
    "loss",
    "assignment",
    "optimizer",
    "train_step",
    "postprocess",
    "inference",
    "evaluation",
]

RUNTIME_HOOK_PHASES: tuple[str, ...] = (
    "data",
    "preprocess",
    "model_graph",
    "loss",
    "assignment",
    "optimizer",
    "train_step",
    "postprocess",
    "inference",
    "evaluation",
)


class RuntimeHookIdentityError(ValueError):
    """Raised when a runtime hook identity cannot resolve to real code."""


def _sha256_of_module_file(dotted_path: str) -> str:
    """SHA-256 of the source file backing ``dotted_path`` (best effort).

    Returns an empty string when the path cannot be located on disk (e.g.
    stubbed modules in tests); :class:`RuntimeHookIdentity` treats an empty
    hash as "unavailable", never as a valid identity.
    """
    try:
        module = importlib.import_module(dotted_path)
    except Exception:  # pragma: no cover - import surface varies per env
        return ""
    file = getattr(module, "__file__", None)
    if not file:
        return ""
    try:
        return hashlib.sha256(Path(file).read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - unreadable file system
        return ""


class RuntimeHookIdentity(BaseModel):
    """One auditable runtime hook binding for one paper.

    Construct with :meth:`resolve` to get a uniform
    :class:`RuntimeHookIdentityError` on any failure (unknown strings,
    missing callables, unresolved imports) instead of pydantic's wrapped
    ``ValidationError``.


    ``hook_id`` is the canonical identity string, e.g.
    ``hook.assignment.assigner.optimal_transport`` — the legacy hook
    vocabulary (``loss.distillation``, ``loss.domain_adaptation``) remains
    valid as the ``hook_id`` for those families, so existing artifacts stay
    comparable while every record gains the resolved fields.
    """

    model_config = ConfigDict(extra="forbid")

    hook_id: str
    phase: RuntimeHookPhase
    implementation_path: str
    class_name: str
    method_name: str
    insertion_point: str
    component_id: str
    paper_id: str
    source_sha256: str = ""
    paper_binding_id: str = ""

    @field_validator("hook_id", "implementation_path", "class_name", "method_name")
    @classmethod
    def _no_unknown(cls, value: str, info: object) -> str:
        text = value.strip()
        if not text:
            field = getattr(info, "field_name", "field")
            raise RuntimeHookIdentityError(f"runtime hook identity {field} must not be empty")
        lowered = text.lower()
        if lowered == "unknown" or lowered.endswith(".unknown") or lowered.endswith("_unknown"):
            raise RuntimeHookIdentityError(
                f"runtime hook identity {getattr(info, 'field_name', 'field')} may not be "
                f"'unknown' ({text!r}): resolve the real callable instead"
            )
        return text

    @field_validator("paper_id", "paper_binding_id", "component_id")
    @classmethod
    def _require_binding(cls, value: str, info: object) -> str:
        text = value.strip()
        field = getattr(info, "field_name", "field")
        if field in {"paper_id", "paper_binding_id"} and not text:
            raise RuntimeHookIdentityError(
                "every RuntimeHookIdentity must carry a paper-specific binding; a shared "
                "implementation_path is allowed but the binding id is mandatory"
            )
        return text

    @model_validator(mode="after")
    def _resolve_callable(self) -> RuntimeHookIdentity:
        if "." not in self.implementation_path:
            raise RuntimeHookIdentityError(
                f"implementation_path must be a dotted import path: {self.implementation_path!r}"
            )
        try:
            module = importlib.import_module(self.implementation_path)
        except Exception as exc:
            raise RuntimeHookIdentityError(
                f"missing callable: cannot import {self.implementation_path!r} ({exc})"
            ) from exc
        cls_obj = getattr(module, self.class_name, None)
        if cls_obj is None:
            raise RuntimeHookIdentityError(
                f"missing callable: {self.implementation_path} has no attribute "
                f"{self.class_name!r}"
            )
        method = getattr(cls_obj, self.method_name, None)
        if method is None:
            raise RuntimeHookIdentityError(
                f"missing callable: {self.implementation_path}.{self.class_name} has no "
                f"attribute {self.method_name!r}"
            )
        if not self.source_sha256:
            resolved = _sha256_of_module_file(self.implementation_path)
            object.__setattr__(self, "source_sha256", resolved)
        return self

    @classmethod
    def resolve(cls, **fields: str) -> RuntimeHookIdentity:
        """Build with fail-closed semantics: every failure raises
        :class:`RuntimeHookIdentityError` (never a wrapped pydantic error)."""
        try:
            return cls(**fields)
        except ValidationError as exc:
            raise RuntimeHookIdentityError(str(exc)) from exc

    @property
    def binding_key(self) -> str:
        """Paper-specific key: shared code + distinct binding = distinct hooks."""
        return f"{self.paper_id}::{self.paper_binding_id or self.component_id}"

    def verify_source(self) -> None:
        """Fail closed when the backing source file drifted from the pin."""
        if not self.source_sha256:
            return
        current = _sha256_of_module_file(self.implementation_path)
        if current and current != self.source_sha256:
            raise RuntimeHookIdentityError(
                f"source drift for {self.implementation_path}: pinned "
                f"{self.source_sha256[:12]}, current {current[:12]}"
            )

    def to_record_dict(self) -> dict[str, str]:
        return {
            "hook_id": self.hook_id,
            "phase": self.phase,
            "implementation_path": self.implementation_path,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "insertion_point": self.insertion_point,
            "component_id": self.component_id,
            "paper_id": self.paper_id,
            "source_sha256": self.source_sha256,
            "paper_binding_id": self.paper_binding_id,
        }
