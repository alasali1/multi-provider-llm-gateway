"""Decide whether an imported dependency is the genuine installed package.

A stand-in (a lone openai.py on sys.path, a mock injected into sys.modules, ...)
has no installed-distribution metadata whose files contain the imported module,
so it cannot pass this check.
"""

from __future__ import annotations

import importlib.util
import os
from importlib import metadata
from pathlib import Path

from http_lib import http_distribution


def tracked_distributions() -> tuple[str, ...]:
    """The SDKs, tiktoken, pydantic, and whichever HTTP library the SDKs are built on."""
    http = sorted({http_distribution("openai"), http_distribution("anthropic")})
    return ("openai", "anthropic", *http, "tiktoken", "pydantic")


def inspect_dependency(name: str) -> tuple[bool, str]:
    """Return (is_genuine, human-readable detail) for the distribution/module `name`."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return False, "not installed"
        return False, f"importable from {spec.origin} but has NO installed-package metadata (stand-in?)"

    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        return False, f"metadata for {dist.version} found but the module cannot be imported"

    # abspath (not resolve): venvs and uv caches symlink files, and we want the path Python used.
    root = Path(os.path.abspath(dist.locate_file("")))
    origin = Path(os.path.abspath(spec.origin))
    if not origin.is_relative_to(root):
        return False, f"module at {origin} is not inside the installed distribution at {root} (shadowed?)"
    return True, f"{dist.version} at {origin.parent}"
