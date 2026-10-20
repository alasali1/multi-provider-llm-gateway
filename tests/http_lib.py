"""Which HTTP library are the installed SDKs built on?

openai>=3.0 and anthropic>=1.0 use `httpx2` and no longer install `httpx`; older
majors use `httpx`. An http client of the wrong flavour is rejected by the SDKs
(TypeError), so tests must build transports and clients from the matching module.
"""

from __future__ import annotations

import importlib
from importlib import metadata

from packaging.requirements import Requirement

_CANDIDATES = ("httpx2", "httpx")


def http_distribution(sdk: str) -> str:
    """Name of the HTTP library the installed SDK `sdk` depends on ('httpx2' unless it needs 'httpx')."""
    try:
        requires = metadata.requires(sdk) or []
    except metadata.PackageNotFoundError:
        return "httpx2"
    for entry in requires:
        req = Requirement(entry)
        if req.name in _CANDIDATES and not (req.marker and "extra" in str(req.marker)):
            return req.name
    return "httpx2"


def http_module(sdk: str):
    """The importable HTTP module matching the installed SDK ('openai' or 'anthropic')."""
    return importlib.import_module(http_distribution(sdk))
