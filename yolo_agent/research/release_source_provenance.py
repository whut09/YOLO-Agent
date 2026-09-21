"""Live adapter source provenance for the training release (Prompt-18H).

The release previously pinned registry/preflight/acceptance artifacts and
the registry's per-paper implementation fingerprints, but nothing forced
the *live adapter Python source* to stay byte-identical after the freeze:
editing an adapter, rebuilding nothing, and leaving the release commit an
ancestor of HEAD still verified.  This module closes that hole.

Two hash surfaces, both derived from the current repository:

``adapter_source_hashes``
    implementation identity (``<implementation_path>#<adapter_class>``)
    -> :func:`yolo_agent.components.maturity_registry.adapter_source_hash`.
    The existing algorithm already covers the adapter class plus every
    local base class in its MRO, so it is reused verbatim — no second
    hashing scheme.  Shared primitives dedupe naturally: two component ids
    bound to the same implementation identity produce one entry.

``runtime_dependency_files``
    dotted module -> file SHA-256 for the local modules an adapter depends
    on but that are *not* in its MRO (loss helpers, assignment math, graph
    builders, postprocess kernels).  The dependency set is the transitive
    local-import closure (AST-parsed, ``TYPE_CHECKING`` blocks excluded)
    of the implementation modules, so it is a deterministic function of
    the very files it hashes.

Every collector fails closed: a component without a resolvable
implementation is reported back to the caller, and the release layer turns
that into a lock reason rather than silently pinning nothing.
"""

from __future__ import annotations

import ast
import importlib
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from yolo_agent.components.maturity_registry import adapter_source_hash

YOLON_AGENT_PREFIX = "yolo_agent"
#: Closure depth for runtime dependency discovery.  Depth 2 covers the
#: adapter module's own helpers plus the helpers of its immediate local
#: imports (loss math, assignment solvers, graph builders) without walking
#: the whole package tree.
RUNTIME_DEPENDENCY_DEPTH = 2

#: Runtime modules executed by the preflight/training path that the
#: registry's adapter contracts do not import directly (the distillation
#: mechanism-loss family).  They join the dependency closure as explicit
#: seeds so a loss-helper edit cannot escape the provenance pin.
EXTRA_RUNTIME_SEED_MODULES: tuple[str, ...] = (
    "yolo_agent.components.distillation.mechanism_losses",
    "yolo_agent.components.distillation.paper_mechanism_losses",
    # The DA runtime surface actually executed by the preflight (the branch
    # plugin and its alignment math) lives outside the route contracts' own
    # import graph.
    "yolo_agent.components.adapters.domain_adaptation.branch_runtime",
    "yolo_agent.components.adapters.domain_adaptation.branches",
    "yolo_agent.components.adapters.domain_adaptation.feature_alignment",
)


def implementation_identity(contract: Any) -> str:
    """Deduplicated key for one adapter implementation binding."""
    return f"{contract.implementation_path}#{contract.adapter_class}"


def _module_file(dotted: str) -> Path | None:
    try:
        module = importlib.import_module(dotted)
    except Exception:
        return None
    file = getattr(module, "__file__", None)
    if not file:
        return None
    path = Path(file).resolve()
    return path if path.is_file() else None


def _local_imports(path: Path) -> set[str]:
    """yolo_agent modules imported by ``path`` (runtime imports only).

    Uses a manual traversal so ``if TYPE_CHECKING:`` blocks are skipped
    *with their bodies* — a plain ``ast.walk`` would still yield the
    type-only imports nested inside them.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return set()
    modules: set[str] = set()

    def _is_type_checking(node: ast.If) -> bool:
        test = node.test
        candidates: list[ast.AST] = [test]
        while candidates:
            current = candidates.pop()
            if isinstance(current, ast.Name):
                if current.id == "TYPE_CHECKING":
                    return True
            elif isinstance(current, ast.Attribute):
                if current.attr == "TYPE_CHECKING":
                    return True
                candidates.append(current.value)
            elif isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
                candidates.append(current.func)
        return False

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and _is_type_checking(node):
            return  # skip the block and its body entirely
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(YOLON_AGENT_PREFIX):
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(YOLON_AGENT_PREFIX):
                    modules.add(alias.name)
        for child in ast.iter_child_nodes(node):
            _visit(child)

    _visit(tree)
    return modules


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_dependency_files(
    seed_modules: Iterable[str],
    *,
    max_depth: int = RUNTIME_DEPENDENCY_DEPTH,
) -> dict[str, str]:
    """Hash the transitive local-import closure of ``seed_modules``.

    Returns ``{dotted_module: sha256}`` for every reachable local module
    file (the seeds included).  Modules that cannot be imported are skipped
    rather than guessed — the seeds themselves come from resolvable
    contracts.
    """
    files: dict[str, str] = {}
    seen: set[str] = set()
    frontier = list(dict.fromkeys(seed_modules))
    for _ in range(max(max_depth, 1)):
        next_frontier: list[str] = []
        for module_name in frontier:
            if module_name in seen:
                continue
            seen.add(module_name)
            file = _module_file(module_name)
            if file is None:
                continue
            files[module_name] = _sha256_of_file(file)
            for dependency in sorted(_local_imports(file)):
                if dependency not in seen:
                    next_frontier.append(dependency)
        frontier = next_frontier
    return files


def _component_ids_from_registry(registry_payload: dict[str, Any]) -> list[str]:
    ids: set[str] = set()
    if not isinstance(registry_payload, dict):
        return []
    for record in registry_payload.get("records", []) or []:
        if not isinstance(record, dict):
            continue
        ids.update(str(value) for value in record.get("component_ids") or [])
    return sorted(ids)


def collect_release_source_hashes(
    registry_payload: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Collect the Prompt-18H source provenance for the frozen campaign.

    Returns ``(adapter_source_hashes, runtime_dependency_files,
    unresolvable_component_ids)``.  The first mapping is deduplicated by
    implementation identity; the second covers the non-MRO helper closure;
    the third lists component ids whose contract lacks a resolvable
    implementation (the caller must lock on them).
    """
    component_ids = _component_ids_from_registry(registry_payload)
    if not component_ids:
        return {}, {}, []

    from yolo_agent.research.paper_runtime_preflight import _load_component_contracts

    contracts = _load_component_contracts()
    adapter_hashes: dict[str, str] = {}
    unresolvable: list[str] = []
    seed_modules: set[str] = set()
    for component_id in component_ids:
        contract = contracts.get(component_id)
        if contract is None or not contract.implementation_path or not contract.adapter_class:
            unresolvable.append(component_id)
            continue
        seed_modules.add(contract.implementation_path)
        identity = implementation_identity(contract)
        if identity in adapter_hashes:
            continue  # shared primitive: one hash per implementation identity
        try:
            adapter_hashes[identity] = adapter_source_hash(contract)
        except (ValueError, OSError, TypeError, ImportError) as exc:
            raise ReleaseSourceProvenanceError(
                f"adapter source hash unavailable for {component_id}: {exc}"
            ) from exc
    dependencies = (
        runtime_dependency_files(sorted(seed_modules | set(EXTRA_RUNTIME_SEED_MODULES)))
        if seed_modules
        else {}
    )
    return adapter_hashes, dependencies, sorted(unresolvable)


class ReleaseSourceProvenanceError(RuntimeError):
    """Raised when adapter source provenance cannot be computed."""
