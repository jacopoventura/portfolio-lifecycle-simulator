"""Import-smoke test: every importable module in the project must import cleanly.

Catches SyntaxError, ImportError, NameError at the top level — bugs that survive
in lazily-imported files (like the `compute_joint_probability` import that broke
after §4.6 removal) until they're actually used.
"""
import importlib
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Modules with side effects on import (require network, GUI, etc.) — skip them here.
# Each is exercised by its own targeted test elsewhere.
_SKIP = {
    # main_cagr_analysis pulls in yfinance / network on import of some submodules
    # if any are added later; reserved for future use.
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(_PROJECT_ROOT).with_suffix("")
    return ".".join(rel.parts)


def _discover_modules() -> list[str]:
    modules: list[str] = []
    for py in _PROJECT_ROOT.rglob("*.py"):
        # Skip caches, tests, dot-dirs, and venvs
        if any(part.startswith(".") or part in {"__pycache__", "tests", "venv", ".venv"}
               for part in py.relative_to(_PROJECT_ROOT).parts):
            continue
        if py.name.startswith("_"):
            continue
        if py.name == "__init__.py":
            continue
        modules.append(_module_name(py))
    return sorted(modules)


_MODULES = _discover_modules()


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports(module_name: str) -> None:
    """Every project module imports without raising."""
    if module_name in _SKIP:
        pytest.skip(f"explicitly skipped: {module_name}")
    importlib.import_module(module_name)
