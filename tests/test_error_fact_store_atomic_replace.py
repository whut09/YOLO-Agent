"""ErrorFactStore atomic replace survives transient Windows file locks."""

from __future__ import annotations

from pathlib import Path

import pytest

from yolo_agent.core.error_facts import ErrorFactStore, _replace_with_retry


def test_replace_retries_transient_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WinError 5 from a transient lock clears on a later attempt."""
    target = tmp_path / "error_facts_by_node.jsonl"
    target.write_text("seed\n", encoding="utf-8")
    staged = tmp_path / "error_facts_by_node.jsonl.tmp"
    staged.write_text("replacement\n", encoding="utf-8")

    real_replace = Path.replace
    attempts = {"count": 0}

    def flaky_replace(self: Path, target: Path) -> Path:
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise PermissionError(5, "拒绝访问。")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(
        "yolo_agent.core.error_facts.time.sleep", lambda _seconds: None
    )

    _replace_with_retry(staged, target)

    assert attempts["count"] == 3
    assert target.read_text(encoding="utf-8") == "replacement\n"
    assert not staged.exists()


def test_replace_fails_closed_after_exhausting_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "error_facts_by_node.jsonl"
    target.write_text("seed\n", encoding="utf-8")
    staged = tmp_path / "error_facts_by_node.jsonl.tmp"
    staged.write_text("replacement\n", encoding="utf-8")

    def locked_replace(self: Path, target: Path) -> Path:
        raise PermissionError(5, "拒绝访问。")

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(
        "yolo_agent.core.error_facts.time.sleep", lambda _seconds: None
    )

    with pytest.raises(PermissionError):
        _replace_with_retry(staged, target)

    # The destination is untouched and the staged payload survives for forensics.
    assert target.read_text(encoding="utf-8") == "seed\n"
    assert staged.read_text(encoding="utf-8") == "replacement\n"


def test_replace_current_node_uses_the_retrying_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """replace_current_node routes its atomic swap through the retry helper."""
    store = ErrorFactStore(tmp_path)
    calls = {"count": 0}

    def spy_replace(temp_path: Path, path: Path) -> None:
        calls["count"] += 1
        Path.replace(temp_path, path)

    monkeypatch.setattr(
        "yolo_agent.core.error_facts._replace_with_retry", spy_replace
    )

    store.replace_current_node(
        "run-1",
        candidate_id="cand",
        node_id="node",
        protocol_hash="proto",
        facts=[],
    )

    assert calls["count"] == 1
    assert (tmp_path / "run-1" / "error_facts_by_node.jsonl").is_file()
