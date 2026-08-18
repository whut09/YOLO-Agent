"""Certified 83-paper protocol catalog fixture tests. No GPU training."""

from __future__ import annotations

from pathlib import Path

import yaml

from yolo_agent.research.paper_protocol_catalog import (
    certified_paper_ids,
    load_certified_paper_protocols,
    requires_explicit_protocol,
)


def test_hash_fixture_matches_live_catalog() -> None:
    payload = yaml.safe_load(
        Path("tests/fixtures/paper_protocol_hashes.yaml").read_text(encoding="utf-8")
    )
    contracts = {item.paper_id: item.protocol_hash for item in load_certified_paper_protocols()}
    assert payload["count"] == 83
    assert len(payload["papers"]) == 83
    assert {row["paper_id"] for row in payload["papers"]} == set(certified_paper_ids())
    for row in payload["papers"]:
        assert contracts[row["paper_id"]] == row["protocol_hash"]


def test_requires_explicit_protocol_for_catalog_prefixes() -> None:
    assert requires_explicit_protocol("arxiv:2103.14259") is True
    assert requires_explicit_protocol("paper-dummy") is False
