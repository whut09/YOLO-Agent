"""Real teacher exponential-moving-average update primitive.

Mean-teacher distillation and semi-supervised domain adaptation require the
teacher to track the student through an exponential moving average instead of
sharing weights.  This module implements that update over a real
``torch.nn.Module``: parameters *and* floating buffers are averaged, integer
buffers (for example ``num_batches_tracked``) are copied, teacher weights stay
frozen, and the teacher remains in eval mode so normalization statistics are
not polluted.  No training loop lives here.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


class TeacherEmaError(ValueError):
    """Raised when the EMA teacher contract is violated."""


class TeacherEmaUpdater:
    """Maintain a frozen EMA copy of a student module.

    The updater owns a deep copy of ``model`` (the teacher).  ``update``
    applies ``teacher = decay * teacher + (1 - decay) * student`` to every
    floating parameter and floating buffer, copies integer buffers verbatim,
    and never lets gradients reach the teacher.
    """

    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise TeacherEmaError(f"EMA decay must be in [0, 1), got {decay}")
        if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, torch.nn.parallel.DistributedDataParallel
        ):
            raise TeacherEmaError("unwrap the student module before EMA tracking")
        self.decay = float(decay)
        self._teacher = copy.deepcopy(model)
        self._teacher.eval()
        for parameter in self._teacher.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            for buffer in self._teacher.buffers():
                buffer.copy_(buffer)

    @property
    def teacher(self) -> torch.nn.Module:
        """Return the frozen EMA teacher module."""

        return self._teacher

    @torch.no_grad()
    def update(self, student: torch.nn.Module) -> None:
        """Fold one student observation into the EMA teacher."""

        teacher_state = self._teacher.state_dict()
        student_state = student.state_dict()
        missing = teacher_state.keys() - student_state.keys()
        extra = student_state.keys() - teacher_state.keys()
        if missing or extra:
            raise TeacherEmaError(
                "student and EMA teacher state keys diverge: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key]
            if not torch.is_floating_point(teacher_value):
                teacher_value.copy_(student_value)
                continue
            teacher_value.mul_(self.decay).add_(
                student_value.detach().to(teacher_value.dtype), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def restore_backup(self) -> None:
        """No-op placeholder kept for API symmetry with eval-time swapping."""

        return None

    def state_dict(self) -> dict[str, Any]:
        """Return the EMA teacher state for checkpoint persistence."""

        return {
            "decay": self.decay,
            "teacher": copy.deepcopy(self._teacher.state_dict()),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a persisted EMA teacher state exactly."""

        if "decay" not in state or "teacher" not in state:
            raise TeacherEmaError("EMA state requires 'decay' and 'teacher' entries")
        decay = float(state["decay"])
        if not 0.0 <= decay < 1.0:
            raise TeacherEmaError(f"persisted EMA decay is invalid: {decay}")
        self.decay = decay
        self._teacher.load_state_dict(state["teacher"])
        self._teacher.eval()
        for parameter in self._teacher.parameters():
            parameter.requires_grad_(False)


__all__ = ["TeacherEmaError", "TeacherEmaUpdater"]
