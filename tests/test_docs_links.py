"""Relative markdown links across README files and docs/ must resolve."""

from __future__ import annotations

import re

from tests.docs_inventory import DOC_PATHS, REPO_ROOT, read_doc

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SKIPPED_PREFIXES = ("http://", "https://", "mailto:", "#")


def test_relative_links_resolve() -> None:
    broken: list[str] = []
    for path in DOC_PATHS:
        for target in LINK_RE.findall(read_doc(path)):
            if target.startswith(SKIPPED_PREFIXES):
                continue
            clean = target.split("#", 1)[0].strip()
            if not clean:
                continue  # pure in-page anchor
            resolved = (path.parent / clean).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(REPO_ROOT)} -> {target}")
    assert not broken, "broken relative links:\n" + "\n".join(sorted(broken))
