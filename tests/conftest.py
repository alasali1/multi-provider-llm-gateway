import pytest

from dependency_check import inspect_dependency, tracked_distributions


@pytest.fixture(autouse=True)
def _no_real_credentials(request, monkeypatch):
    """The suite must never pick up (or spend) a developer's real API keys."""
    if request.node.get_closest_marker("live"):
        return
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Real-dependency provenance
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def real_deps():
    """Fail (never skip) when a `real_deps` test would run against missing or stand-in packages."""
    problems = []
    for name in tracked_distributions():
        genuine, detail = inspect_dependency(name)
        if not genuine:
            problems.append(f"  {name}: {detail}")
    if problems:
        pytest.fail(
            "real_deps tests need the genuine packages, but:\n" + "\n".join(problems)
            + "\nInstall them with `uv sync` and re-run.",
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _real_deps_tests_require_genuine_packages(request):
    if request.node.get_closest_marker("real_deps"):
        request.getfixturevalue("real_deps")


def pytest_report_header(config):
    lines = ["dependency provenance:"]
    for name in tracked_distributions():
        genuine, detail = inspect_dependency(name)
        lines.append(f"  {'REAL    ' if genuine else 'NOT REAL'} {name}: {detail}")
    return lines


def _bucket(report) -> str:
    keywords = getattr(report, "keywords", {}) or {}
    if "real_deps" in keywords or "test_real_dependencies" in report.nodeid:
        return "REAL-DEPENDENCY tests (genuine SDKs + httpx.MockTransport, no live API)"
    if "test_dependency_declarations" in report.nodeid:
        return "STATIC tests (read project files; no SDKs, no mocks)"
    return "UNIT tests (fake provider clients; these do NOT prove SDK compatibility)"


def pytest_terminal_summary(terminalreporter):
    counts: dict[str, dict[str, int]] = {}
    for outcome in ("passed", "failed", "error", "skipped"):
        for report in terminalreporter.stats.get(outcome, []):
            bucket = counts.setdefault(_bucket(report), {})
            bucket[outcome] = bucket.get(outcome, 0) + 1
    if not counts:
        return
    terminalreporter.section("test provenance")
    for name in sorted(counts):
        detail = ", ".join(f"{n} {outcome}" for outcome, n in counts[name].items())
        terminalreporter.write_line(f"{name}: {detail}")
