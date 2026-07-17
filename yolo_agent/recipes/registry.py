"""Versioned local registry for component recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from yolo_agent.components.contracts import ComponentContract
from yolo_agent.recipes.schemas import RecipeSpec, recipe_from_mapping


class RecipeRegistry:
    """In-memory registry with version-aware recipe lookup."""

    def __init__(self, recipes: Iterable[RecipeSpec] = (), component_contracts: Iterable[ComponentContract] = ()) -> None:
        self._recipes: dict[tuple[str, str], RecipeSpec] = {}
        self._components = {item.component_id: item for item in component_contracts}
        for recipe in recipes:
            self.register(recipe)

    @classmethod
    def from_path(cls, path: Path | str, *, component_contracts: Iterable[ComponentContract] = ()) -> "RecipeRegistry":
        source = Path(path)
        with source.open("r", encoding="utf-8-sig") as file:
            raw = yaml.safe_load(file) or {}
        entries = raw.get("recipes", raw) if isinstance(raw, dict) else []
        if isinstance(entries, dict) and "recipe_id" in entries:
            entries = [entries]
        if not isinstance(entries, list):
            raise ValueError(f"Recipe YAML must contain a recipes list: {source}")
        return cls((recipe_from_mapping(item) for item in entries if isinstance(item, dict)), component_contracts=component_contracts)

    def register(self, recipe: RecipeSpec) -> None:
        if self._components:
            recipe.validate_components(self._components)
        self._recipes[(recipe.recipe_id, recipe.version)] = recipe

    def get(self, recipe_id: str, version: str | None = None) -> RecipeSpec | None:
        matches = [recipe for (rid, _), recipe in self._recipes.items() if rid == recipe_id]
        if not matches:
            return None
        return self._recipes.get((recipe_id, version)) if version else max(matches, key=lambda item: _version_key(item.version))

    def list(self, *, maturity: str | None = None, executable_only: bool = False) -> list[RecipeSpec]:
        recipes = list(self._recipes.values())
        if maturity is not None:
            recipes = [recipe for recipe in recipes if recipe.maturity == maturity]
        if executable_only:
            recipes = [recipe for recipe in recipes if recipe.is_executable]
        return sorted(recipes, key=lambda item: (item.recipe_id, _version_key(item.version)))

    def query(self, *, target_metric: str | None = None, target_error_fact: str | None = None, component_id: str | None = None, executable_only: bool = False) -> list[RecipeSpec]:
        recipes = self.list(executable_only=executable_only)
        if target_metric:
            recipes = [item for item in recipes if target_metric in item.target_metrics]
        if target_error_fact:
            recipes = [item for item in recipes if any(target_error_fact in str(fact) for fact in item.target_error_facts)]
        if component_id:
            recipes = [item for item in recipes if component_id in item.component_ids]
        return recipes


def _version_key(version: str) -> tuple[int, ...] | tuple[str]:
    try:
        return tuple(int(part) for part in version.lstrip("v").split("."))
    except ValueError:
        return (version,)


__all__ = ["RecipeRegistry"]
