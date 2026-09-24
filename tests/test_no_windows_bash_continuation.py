"""Windows examples must be single-line, and known path typos must stay out."""

from __future__ import annotations

from tests.docs_inventory import DOC_PATHS, REPO_ROOT, iter_fenced_blocks, read_doc

FORBIDDEN_TYPO = "datatset"
CONTINUATION_CHARS = ("\\", "`")
TYPO_EXEMPT = {REPO_ROOT / "docs" / "DOCUMENTATION_AUDIT.md"}


def test_powershell_blocks_have_no_line_continuation() -> None:
    offenders: list[str] = []
    for path in DOC_PATHS:
        for language, block in iter_fenced_blocks(read_doc(path)):
            if language not in {"powershell", "pwsh", "ps1"}:
                continue
            for offset, line in enumerate(block, 1):
                stripped = line.rstrip()
                if stripped.endswith(CONTINUATION_CHARS):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: fenced line {offset}")
    assert not offenders, "line continuations in Windows examples:\n" + "\n".join(offenders)


def test_no_known_path_typos() -> None:
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in DOC_PATHS
        if path not in TYPO_EXEMPT and FORBIDDEN_TYPO in read_doc(path)
    ]
    assert not offenders, f"typo {FORBIDDEN_TYPO!r} found in: {offenders}"
