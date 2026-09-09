"""Freeze the historical README-certified paper campaign by exact identity.

This module is an offline audit utility.  It never builds a trainer, probes a
GPU, or derives campaign membership from current runtime maturity.
"""

from __future__ import annotations

import re
from pathlib import Path

from yolo_agent.research.paper_83_campaign_schemas import (
    PAPER_83_COUNT,
    PAPER_85_COUNT,
    ReadmeCoverageDeclaration,
)


class Paper83CampaignError(ValueError):
    """Raised when the historical paper-83 source cannot be verified."""


_RATIO_ROW = re.compile(
    r"\|\s*{label}\s*\|\s*(?P<numerator>\d+)\s*/\s*(?P<denominator>\d+)\b",
)
_HASH_LINE = re.compile(
    r"(?P<label>Audit snapshot|Acceptance hash)\s*:\s*`(?P<hash>[0-9a-f]{{64}})`",
    re.IGNORECASE,
)


def parse_readme_coverage(readme_path: Path | str) -> ReadmeCoverageDeclaration:
    """Parse coverage numbers and lineage hashes from the current README."""

    path = Path(readme_path)
    if not path.is_file():
        raise Paper83CampaignError(f"README_NOT_FOUND={path}")
    text = path.read_text(encoding="utf-8-sig")
    method = _parse_ratio_row(text, "Compatible papers with valid MethodProfile")
    certified = _parse_ratio_row(text, "Compatible papers reusing a certified adapter")
    hashes = {
        match.group("label").casefold(): match.group("hash")
        for match in _HASH_LINE.finditer(text)
    }
    try:
        audit_snapshot_hash = hashes["audit snapshot"]
        acceptance_hash = hashes["acceptance hash"]
    except KeyError as exc:
        raise Paper83CampaignError(
            "README lineage hashes are incomplete; expected Audit snapshot and Acceptance hash"
        ) from exc
    return ReadmeCoverageDeclaration(
        method_profile_numerator=method[0],
        method_profile_denominator=method[1],
        certified_adapter_numerator=certified[0],
        certified_adapter_denominator=certified[1],
        audit_snapshot_hash=audit_snapshot_hash,
        acceptance_hash=acceptance_hash,
    )


def assert_readme_campaign_shape(
    declaration: ReadmeCoverageDeclaration,
) -> None:
    """Fail closed unless README still declares the requested 83/85 campaign."""

    if (
        declaration.certified_adapter_numerator != PAPER_83_COUNT
        or declaration.certified_adapter_denominator != PAPER_85_COUNT
        or declaration.method_profile_numerator != PAPER_85_COUNT
        or declaration.method_profile_denominator != PAPER_85_COUNT
    ):
        raise Paper83CampaignError(
            "README campaign shape changed: "
            f"README_EXPECTED_CERTIFIED={PAPER_83_COUNT} "
            f"README_ACTUAL_CERTIFIED={declaration.certified_adapter_numerator} "
            f"README_COMPATIBLE_DENOMINATOR={declaration.certified_adapter_denominator}"
        )


def _parse_ratio_row(text: str, label: str) -> tuple[int, int]:
    pattern = re.compile(_RATIO_ROW.pattern.format(label=re.escape(label)))
    match = pattern.search(text)
    if match is None:
        raise Paper83CampaignError(f"README_COVERAGE_ROW_NOT_FOUND={label}")
    return int(match.group("numerator")), int(match.group("denominator"))


__all__ = [
    "Paper83CampaignError",
    "assert_readme_campaign_shape",
    "parse_readme_coverage",
]
