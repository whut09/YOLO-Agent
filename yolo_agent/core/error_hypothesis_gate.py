"""Prompt-18I-v2 §3: hypotheses may enter a training candidate only with evidence.

The gate binds the attribution layer to the identity layer: a hypothesis
(EvidenceLinkedHypothesis from the attribution module, or any object
carrying ``evidence_fact_ids``) may back a training candidate only when
every fact id it cites exists among the facts extracted from the round's
real DetectionErrorProfile.  Hypotheses without evidence — or citing
facts that were never measured — are rejected *before* candidate
registration, so an unevidenced story can never reach the GPU queue.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yolo_agent.core.error_fact_identity import ErrorFactIdentity

GATE_SCHEMA_VERSION = "error_hypothesis_gate.v1"


class HypothesisAdmission(BaseModel):
    """Admission verdict for one hypothesis against real facts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = GATE_SCHEMA_VERSION
    hypothesis_id: str
    admitted: bool
    missing_fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def validate_admission(self) -> "HypothesisAdmission":
        if self.admitted and self.missing_fact_ids:
            raise ValueError("an admitted hypothesis cannot miss fact ids")
        return self


def _evidence_fact_ids(hypothesis: object) -> list[str]:
    """Read the fact ids a hypothesis cites, whichever shape it has."""
    ids: list[str] = []
    evidence = getattr(hypothesis, "evidence", None)
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        for link in evidence:
            link_ids = getattr(link, "fact_ids", None)
            if isinstance(link_ids, (list, tuple)):
                ids.extend(str(fid) for fid in link_ids)
    direct = getattr(hypothesis, "evidence_fact_ids", None)
    if isinstance(direct, (list, tuple)):
        ids.extend(str(fid) for fid in direct)
    return ids


def admit_hypothesis(
    hypothesis: object,
    facts: Sequence[ErrorFactIdentity],
) -> HypothesisAdmission:
    """Admit a hypothesis only when every cited fact id is real."""
    hypothesis_id = str(getattr(hypothesis, "hypothesis_id", "") or "<unnamed>")
    cited = _evidence_fact_ids(hypothesis)
    known = {fact.error_fact_id for fact in facts}
    missing = sorted({fid for fid in cited if fid not in known})
    if not cited:
        return HypothesisAdmission(
            hypothesis_id=hypothesis_id,
            admitted=False,
            reason="hypothesis cites no evidence fact ids",
        )
    if missing:
        return HypothesisAdmission(
            hypothesis_id=hypothesis_id,
            admitted=False,
            missing_fact_ids=missing,
            reason=f"hypothesis cites facts that were never measured: {missing}",
        )
    return HypothesisAdmission(hypothesis_id=hypothesis_id, admitted=True)


def admit_hypotheses(
    hypotheses: Sequence[object],
    facts: Sequence[ErrorFactIdentity],
) -> list[HypothesisAdmission]:
    """Admission verdicts for a whole round's hypotheses."""
    return [admit_hypothesis(h, facts) for h in hypotheses]


def training_candidate_allowed(
    admissions: Sequence[HypothesisAdmission],
) -> bool:
    """A training candidate may be backed only by admitted hypotheses.

    A candidate must rest on at least one admitted hypothesis; any
    rejected hypothesis disqualifies the batch outright.
    """
    return bool(admissions) and all(item.admitted for item in admissions)


__all__ = [
    "GATE_SCHEMA_VERSION",
    "HypothesisAdmission",
    "admit_hypotheses",
    "admit_hypothesis",
    "training_candidate_allowed",
]
