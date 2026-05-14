"""Import smoke tests — every non-test module must parse and import cleanly.

Catches SyntaxErrors and ImportErrors in lazily-imported modules that unit tests
do not exercise (e.g. reports/section4.py before the decumulation pipeline runs).
"""
from __future__ import annotations

import importlib
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SKIP_DIRS = {".venv", "venv", "tests", "__pycache__", "historical_data", ".git"}


def _discover_modules() -> list[str]:
    modules: list[str] = []
    for path in _ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in path.relative_to(_ROOT).parts):
            continue
        if path.name == "__init__.py":
            rel = path.parent.relative_to(_ROOT)
            if rel == pathlib.Path("."):
                continue
            modules.append(rel.as_posix().replace("/", "."))
        else:
            rel = path.with_suffix("").relative_to(_ROOT)
            modules.append(rel.as_posix().replace("/", "."))
    return sorted(modules)


_MODULES = _discover_modules()


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
