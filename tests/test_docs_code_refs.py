"""Backtick `package.module.Symbol` references in docs must exist in code."""

from __future__ import annotations

import importlib
import re

from tests.docs_inventory import DOC_PATHS, REPO_ROOT, read_doc

REF_RE = re.compile(r"`([a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+\.[A-Z][A-Za-z0-9_]*)`")


def test_documented_code_references_exist() -> None:
    problems: list[str] = []
    for path in DOC_PATHS:
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(read_doc(path).splitlines(), 1):
            if "conceptual" in line.lower():
                continue
            for reference in REF_RE.findall(line):
                module_name, _, symbol = reference.rpartition(".")
                try:
                    module = importlib.import_module(module_name)
                    getattr(module, symbol)
                except (ImportError, AttributeError) as exc:
                    problems.append(f"{rel}:{lineno}: {reference} ({exc})")
    assert not problems, "unresolvable code references:\n" + "\n".join(problems)
