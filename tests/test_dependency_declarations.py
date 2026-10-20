"""Static checks: real packages are declared and used, and stand-ins can't hide in the repo.

These read project files only. They need neither the SDKs nor any mocks.
"""

from __future__ import annotations

import ast
import os
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_MODULES = ("gateway.py", "pricing.py", "schemas.py", "app.py")
REQUIRED_RUNTIME = ("openai", "anthropic", "tiktoken", "pydantic")
REQUIRED_DEV = ("pytest", "httpx2")  # httpx2 is imported directly by the tests (SDKs use it for transports)
SDK_NAMES = {"openai", "anthropic", "httpx", "httpx2", "tiktoken"}


@pytest.fixture(scope="module")
def src() -> Path:
    for candidate in (ROOT, ROOT / "src"):
        if (candidate / "gateway.py").exists():
            return candidate
    pytest.fail("gateway.py not found in the project root or src/", pytrace=False)


def _requirements(entries) -> dict[str, Requirement]:
    reqs = (Requirement(e) for e in entries if isinstance(e, str))
    return {canonicalize_name(r.name): r for r in reqs}


def _imported_top_level_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_real_packages_are_declared_with_version_floors():
    pyproject = ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml is missing from the project root"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    deps = _requirements(data["project"]["dependencies"])
    dev = _requirements(data.get("dependency-groups", {}).get("dev", []))
    for group, names in ((deps, REQUIRED_RUNTIME), (dev, REQUIRED_DEV)):
        for name in names:
            assert name in group, f"{name} is not declared in the right dependency list"
            spec = str(group[name].specifier)
            assert ">=" in spec, f"{name} needs a minimum-version constraint (found {spec!r})"
    assert data["project"]["requires-python"].replace(" ", "").startswith(">=3.11")


def test_gateway_imports_the_real_sdks(src):
    imported = _imported_top_level_modules(src / "gateway.py")
    assert {"openai", "anthropic", "tiktoken"} <= imported


def test_production_modules_contain_no_test_doubles(src):
    forbidden = {"unittest", "mock", "pytest", "fake_providers", "conftest", "tests"}
    for name in PRODUCTION_MODULES:
        path = src / name
        if path.exists():
            assert not (_imported_top_level_modules(path) & forbidden), f"{name} imports test tooling"


def test_project_tree_contains_no_standin_packages():
    skip = {".venv", "venv", ".git", "node_modules", "__pycache__"}
    offenders = []
    for current, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in (*dirs, *files):
            stem = name[:-3] if name.endswith(".py") else name
            if stem in SDK_NAMES:
                offenders.append(str(Path(current, name).relative_to(ROOT)))
    assert not offenders, f"stand-in modules shadowing real packages: {offenders}"
