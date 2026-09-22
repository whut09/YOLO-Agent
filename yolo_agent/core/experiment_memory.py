"""Experiment memory: no infinite repeats of failed experiments (§4).

Every attempted experiment is remembered by a four-part fingerprint:

* ``problem_fingerprint`` — the error-structure hash of the round's profile
  (which error surfaces were open, not the round id),
* ``action_fingerprint`` — the chosen ``action_id``,
* ``parameter_fingerprint`` — the resolved parameter assignment,
* ``parent_fingerprint`` — the parent checkpoint/config identity the action
  ran from.

A repeat request with the same tuple whose previous outcome was a failure
(``rejected`` / ``rolled_back`` / ``refined_away``) is refused; only an
explicitly ``retry_allowed`` reset (e.g. after the underlying component or
dataset changed) can clear it.  Successful outcomes are remembered too so
the loop can avoid re-running promotions needlessly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from yolo_agent.core.detection_error_profile import DetectionErrorProfile

MEMORY_SCHEMA_VERSION = "experiment_memory.v1"

Outcome = Literal[
    "promoted",
    "refined",
    "rejected",
    "rolled_back",
    "failed",
]

#: Outcomes that make an identical repeat pointless.
BLOCKED_OUTCOMES: frozenset[str] = frozenset(
    {"rejected", "rolled_back", "failed"}
)


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def problem_fingerprint(profile: DetectionErrorProfile) -> str:
    """Hash the open error structure of a profile — stable across runs.

    Round ids and run ids are excluded: the same error structure met by the
    same experiment from the same parent must not repeat, regardless of what
    the loop calls the round.
    """
    return _stable_hash(
        {
            "global": profile.global_.model_dump(mode="json"),
            "scale": profile.scale.model_dump(mode="json"),
            "false_negative": profile.false_negative.model_dump(mode="json"),
            "false_positive": profile.false_positive.model_dump(mode="json"),
            "localization": profile.localization.model_dump(mode="json"),
            "classification": profile.classification.model_dump(mode="json"),
        }
    )


class ExperimentMemoryRecord(BaseModel):
    """One remembered experiment attempt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = MEMORY_SCHEMA_VERSION
    problem_fingerprint: str
    action_id: str
    parameter_fingerprint: str
    parent_fingerprint: str
    outcome: Outcome
    candidate_id: str
    detail: str = ""

    @property
    def memory_key(self) -> str:
        return _stable_hash(
            {
                "problem": self.problem_fingerprint,
                "action": self.action_id,
                "parameters": self.parameter_fingerprint,
                "parent": self.parent_fingerprint,
            }
        )


class ExperimentMemory:
    """In-memory + on-disk store of attempted experiments."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: dict[str, ExperimentMemoryRecord] = {}
        if self._path is not None and self._path.is_file():
            import yaml

            payload = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            for raw in payload.get("records", []):
                record = ExperimentMemoryRecord.model_validate(raw)
                self._records[record.memory_key] = record

    def record(
        self,
        *,
        profile: DetectionErrorProfile,
        action_id: str,
        parameters: dict,
        parent_fingerprint: str,
        outcome: Outcome,
        candidate_id: str,
        detail: str = "",
    ) -> ExperimentMemoryRecord:
        record = ExperimentMemoryRecord(
            problem_fingerprint=problem_fingerprint(profile),
            action_id=action_id,
            parameter_fingerprint=_stable_hash(parameters),
            parent_fingerprint=parent_fingerprint,
            outcome=outcome,
            candidate_id=candidate_id,
            detail=detail,
        )
        self._records[record.memory_key] = record
        self._flush()
        return record

    def check_repeat(
        self,
        *,
        profile: DetectionErrorProfile,
        action_id: str,
        parameters: dict,
        parent_fingerprint: str,
    ) -> ExperimentMemoryRecord | None:
        """Return the blocking record if this exact experiment already failed.

        ``None`` means the experiment is new (or previously succeeded in a
        way that does not forbid a rerun).
        """
        key = _stable_hash(
            {
                "problem": problem_fingerprint(profile),
                "action": action_id,
                "parameters": _stable_hash(parameters),
                "parent": parent_fingerprint,
            }
        )
        previous = self._records.get(key)
        if previous is not None and previous.outcome in BLOCKED_OUTCOMES:
            return previous
        return None

    def reset(
        self,
        *,
        profile: DetectionErrorProfile,
        action_id: str,
        parameters: dict,
        parent_fingerprint: str,
    ) -> bool:
        """Explicitly allow retrying a failed experiment (e.g. after a fix)."""
        key = _stable_hash(
            {
                "problem": problem_fingerprint(profile),
                "action": action_id,
                "parameters": _stable_hash(parameters),
                "parent": parent_fingerprint,
            }
        )
        return self._records.pop(key, None) is not None

    def _flush(self) -> None:
        if self._path is None:
            return
        import yaml

        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "records": [
                record.model_dump(mode="json")
                for record in sorted(
                    self._records.values(), key=lambda item: item.memory_key
                )
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


__all__ = [
    "BLOCKED_OUTCOMES",
    "ExperimentMemory",
    "ExperimentMemoryRecord",
    "Outcome",
    "problem_fingerprint",
]
