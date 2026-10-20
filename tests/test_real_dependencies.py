"""Compatibility tests against the REAL openai / anthropic / HTTP library / tiktoken packages.

No live API calls and no API keys: HTTP is served by an in-process MockTransport, so the
genuine SDKs build genuine requests and parse genuine wire-format responses
(SSE streams, error bodies, headers). The gateway code under test is unmodified.

The transport comes from whichever HTTP library the installed SDKs are built on
(httpx2 for openai>=3 / anthropic>=1, httpx for older majors); see http_lib.py.

Run only these:   uv run pytest -m real_deps
If a dependency is missing or is a stand-in, these tests ERROR (they never skip),
so a green result really does mean "verified against the real packages".
"""

from __future__ import annotations

import inspect
import json
import logging
import tomllib
import traceback
from importlib import metadata
from pathlib import Path

import anthropic
import openai
import pytest
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import gateway as gateway_module
from fake_providers import EXPECTED_TEXT, TOKENS, run_stream
from gateway import UnifiedLLMGateway
from http_lib import http_distribution, http_module
from schemas import (
    AuthenticationError,
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
)

pytestmark = pytest.mark.real_deps

PROVIDERS = pytest.mark.parametrize("provider", ["openai", "anthropic"])
KEYS = {"openai": "sk-test-" + "C" * 24, "anthropic": "sk-ant-test-" + "D" * 24}
MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}
INPUT_TOKENS, OUTPUT_TOKENS = 120, 40


# --------------------------------------------------------------------------- #
# Wire-format fixtures (what the real APIs send over SSE)
# --------------------------------------------------------------------------- #
def _openai_sse(tokens, prompt_tokens, completion_tokens) -> bytes:
    def chunk(delta, finish=None, usage=None, has_choice=True):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if has_choice else [],
            "usage": usage,
        }

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    chunks = [
        chunk({"role": "assistant", "content": ""}),
        *(chunk({"content": t}) for t in tokens),
        chunk({}, finish="stop"),
        chunk(None, usage=usage, has_choice=False),  # include_usage: final chunk, empty choices
    ]
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def _anthropic_sse(tokens, input_tokens, output_tokens) -> bytes:
    message = {
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": "claude-haiku-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 1},
    }
    events = [
        ("message_start", {"type": "message_start", "message": message}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("ping", {"type": "ping"}),
        *(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": t}}) for t in tokens),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": output_tokens}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()


def _success_body(provider: str) -> bytes:
    if provider == "openai":
        return _openai_sse(TOKENS, INPUT_TOKENS, OUTPUT_TOKENS)
    return _anthropic_sse(TOKENS, INPUT_TOKENS, OUTPUT_TOKENS)


def _error_payload(provider: str, message: str) -> dict:
    if provider == "openai":
        return {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}}
    return {"type": "error", "error": {"type": "api_error", "message": message}}


REQUEST_ID_HEADER = {"openai": "x-request-id", "anthropic": "request-id"}


# --------------------------------------------------------------------------- #
# Real SDK clients wired to an in-process transport.
# Each factory below takes the HTTP module (httpx2 or httpx) matching the installed SDK.
# --------------------------------------------------------------------------- #
def _sse(body: bytes, seen: list):
    def build(lib):
        def handler(request):
            seen.append(request)
            return lib.Response(200, headers={"content-type": "text/event-stream"}, content=body)

        return lib.MockTransport(handler)

    return build


def _http_error(status: int, payload: dict, headers: dict | None = None):
    def build(lib):
        return lib.MockTransport(lambda request: lib.Response(status, json=payload, headers=headers or {}))

    return build


def _transport_failure(make_exc):
    def build(lib):
        def handler(request):
            raise make_exc(lib)

        return lib.MockTransport(handler)

    return build


def _gateway(provider: str, transport_factory) -> tuple[UnifiedLLMGateway, str]:
    # The gateway builds its own real clients with its real constructor arguments first...
    gw = UnifiedLLMGateway(openai_api_key=KEYS["openai"], anthropic_api_key=KEYS["anthropic"])
    lib = http_module(provider)
    http_client = lib.AsyncClient(transport=transport_factory(lib))
    # ...then the client under test is swapped for a real one on an in-process transport.
    if provider == "openai":
        gw._openai = AsyncOpenAI(api_key=KEYS["openai"], http_client=http_client, max_retries=0)
    else:
        gw._anthropic = AsyncAnthropic(api_key=KEYS["anthropic"], http_client=http_client, max_retries=0)
    return gw, MODELS[provider]


@pytest.fixture
def offline_prompt_count(monkeypatch):
    """Keeps the SDK round trips independent of the tiktoken download (covered separately below)."""
    monkeypatch.setattr(gateway_module, "_count_openai_prompt_tokens", lambda model, prompt: INPUT_TOKENS)


# --------------------------------------------------------------------------- #
# Declarations vs what is actually installed
# --------------------------------------------------------------------------- #
def test_installed_versions_satisfy_the_constraints_declared_in_pyproject():
    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = [*data["project"]["dependencies"], *data.get("dependency-groups", {}).get("dev", [])]
    wanted = {canonicalize_name(Requirement(e).name): Requirement(e) for e in declared if isinstance(e, str)}
    for name in ("openai", "anthropic", "tiktoken", "pydantic", "httpx2"):
        installed = metadata.version(name)
        assert wanted[name].specifier.contains(installed, prereleases=True), (
            f"installed {name} {installed} violates declared constraint {wanted[name].specifier}"
        )


@pytest.mark.parametrize("which", ["openai", "anthropic", "http_library"])
def test_module_version_matches_installed_metadata(which):
    module = http_module("openai") if which == "http_library" else {"openai": openai, "anthropic": anthropic}[which]
    dist = http_distribution("openai") if which == "http_library" else which
    if not hasattr(module, "__version__"):
        pytest.skip(f"{module.__name__} exposes no __version__")
    assert module.__version__ == metadata.version(dist)


def test_both_sdks_are_built_on_the_same_http_library():
    assert http_distribution("openai") == http_distribution("anthropic")


# --------------------------------------------------------------------------- #
# SDK surface the gateway depends on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sdk", [openai, anthropic], ids=lambda m: m.__name__)
def test_exception_hierarchy_matches_what_the_gateway_assumes(sdk):
    for name in ("APIStatusError", "APIConnectionError", "APITimeoutError",
                 "RateLimitError", "AuthenticationError", "PermissionDeniedError"):
        assert hasattr(sdk, name), f"{sdk.__name__}.{name} is missing"
    for status_error in (sdk.RateLimitError, sdk.AuthenticationError, sdk.PermissionDeniedError):
        assert issubclass(status_error, sdk.APIStatusError)
    # _translate_error tests APITimeoutError before APIConnectionError, so it must be a subclass:
    assert issubclass(sdk.APITimeoutError, sdk.APIConnectionError)
    assert not issubclass(sdk.APIConnectionError, sdk.APIStatusError)


def test_gateway_builds_real_clients_with_its_constructor_arguments():
    gw = UnifiedLLMGateway(openai_api_key=KEYS["openai"], anthropic_api_key=KEYS["anthropic"])
    assert isinstance(gw._openai, AsyncOpenAI)
    assert isinstance(gw._anthropic, AsyncAnthropic)


def test_create_signatures_accept_the_arguments_the_gateway_sends():
    openai_params = set(inspect.signature(AsyncOpenAI(api_key=KEYS["openai"]).chat.completions.create).parameters)
    assert {"model", "messages", "temperature", "stream", "stream_options"} <= openai_params
    anthropic_params = set(inspect.signature(AsyncAnthropic(api_key=KEYS["anthropic"]).messages.create).parameters)
    # anthropic>=1.0 no longer has a `temperature` keyword, and Claude 5.x rejects non-default values.
    assert {"model", "max_tokens", "messages", "stream"} <= anthropic_params


# --------------------------------------------------------------------------- #
# Real SDK streaming round trip (real request building + real SSE parsing)
# --------------------------------------------------------------------------- #
@PROVIDERS
def test_real_sdk_stream_round_trip(provider, offline_prompt_count):
    seen: list = []
    gw, model = _gateway(provider, _sse(_success_body(provider), seen))

    chunks = run_stream(gw, "Say hello", model, temperature=0.3)

    assert [c.delta_text for c in chunks[:-1]] == TOKENS
    assert "".join(c.delta_text for c in chunks) == EXPECTED_TEXT
    telemetry = chunks[-1].telemetry
    assert chunks[-1].is_final and telemetry is not None
    assert (telemetry.input_tokens, telemetry.output_tokens) == (INPUT_TOKENS, OUTPUT_TOKENS)
    assert telemetry.usage_source == "provider"
    assert telemetry.ttft_ms is not None and telemetry.total_cost_usd > 0

    (request,) = seen  # what the real SDK actually put on the wire
    payload = json.loads(request.content)
    assert request.method == "POST"
    assert payload["model"] == model
    assert payload["stream"] is True
    if provider == "openai":
        assert payload["temperature"] == 0.3
    else:
        assert "temperature" not in payload  # Claude 5.x answers 400 to a non-default temperature
    assert payload["messages"] == [{"role": "user", "content": "Say hello"}]
    if provider == "openai":
        assert request.url.path.endswith("/chat/completions")
        assert payload["stream_options"] == {"include_usage": True}
        assert request.headers["authorization"] == f"Bearer {KEYS['openai']}"
    else:
        assert request.url.path.endswith("/messages")
        assert isinstance(payload["max_tokens"], int) and payload["max_tokens"] > 0
        assert request.headers["x-api-key"] == KEYS["anthropic"]
        assert "anthropic-version" in request.headers


# --------------------------------------------------------------------------- #
# Real SDK error handling
# --------------------------------------------------------------------------- #
@PROVIDERS
@pytest.mark.parametrize(
    "status,expected",
    [(401, AuthenticationError), (403, AuthenticationError), (429, RateLimitExceededError), (500, GatewayProviderError)],
    ids=["401-auth", "403-permission", "429-rate-limit", "500-server"],
)
def test_real_sdk_http_errors_map_to_domain_errors(provider, status, expected):
    headers = {REQUEST_ID_HEADER[provider]: "req_test_123", "retry-after": "3"}
    gw, model = _gateway(provider, _http_error(status, _error_payload(provider, "nope"), headers))

    with pytest.raises(GatewayProviderError) as info:
        run_stream(gw, "hi", model)

    err = info.value
    assert type(err) is expected
    assert err.provider == provider
    assert err.status_code == status
    assert err.request_id == "req_test_123"
    if status == 429:
        assert err.retry_after_s == 3.0


@PROVIDERS
@pytest.mark.parametrize("failure", ["connect_error", "read_timeout"])
def test_real_sdk_transport_failures_map_to_connection_errors(provider, failure):
    make_exc = {
        "connect_error": lambda lib: lib.ConnectError("connection refused"),
        "read_timeout": lambda lib: lib.ReadTimeout("read timed out"),
    }[failure]
    gw, model = _gateway(provider, _transport_failure(make_exc))

    with pytest.raises(ProviderConnectionError) as info:
        run_stream(gw, "hi", model)
    assert info.value.provider == provider


def _assert_no_fragment(secret: str, label: str, text: str, window: int = 12) -> None:
    leaked = [secret[i:i + window] for i in range(len(secret) - window + 1) if secret[i:i + window] in text]
    assert not leaked, f"{label} leaks part of the API key"


@PROVIDERS
def test_real_sdk_errors_never_leak_the_key(provider, caplog, capsys):
    secret = KEYS[provider]
    payload = _error_payload(provider, f"Incorrect API key provided: {secret}")
    gw, model = _gateway(provider, _http_error(401, payload))
    caplog.set_level(logging.DEBUG)  # includes the SDKs' own debug logging

    with pytest.raises(AuthenticationError) as info:
        run_stream(gw, "hi", model)

    exc = info.value
    out, err_out = capsys.readouterr()
    for label, text in {
        "str(exception)": str(exc),
        "repr(exception)": repr(exc),
        "exception attributes": repr(vars(exc)),
        "formatted traceback": "".join(traceback.format_exception(exc)),
        "log output": caplog.text,
        "stdout": out,
        "stderr": err_out,
    }.items():
        _assert_no_fragment(secret, label, text)


# --------------------------------------------------------------------------- #
# Real tiktoken (needs the encoding file: one-time download, or a pre-filled TIKTOKEN_CACHE_DIR)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_encoding():
    try:
        return tiktoken.encoding_for_model(MODELS["openai"])
    except Exception as exc:
        pytest.skip(
            f"tiktoken could not load its encoding ({type(exc).__name__}); it needs a one-time download "
            "or a pre-filled TIKTOKEN_CACHE_DIR. These tests did NOT run."
        )


def test_gateway_prompt_estimate_matches_real_tiktoken(real_encoding):
    from test_gateway import PROMPT  # the 50-word prompt used by the unit tests

    expected = len(real_encoding.encode(PROMPT))
    assert real_encoding.name == "o200k_base"
    assert expected >= 70, "prompt too short for a +/-10% tolerance to absorb the chat framing tokens"

    estimate = gateway_module._count_openai_prompt_tokens(MODELS["openai"], PROMPT)
    assert estimate == expected + gateway_module._OPENAI_CHAT_OVERHEAD
    assert estimate == pytest.approx(expected, rel=0.10)
    # OpenAI adds ~7 framing tokens to a one-message chat prompt; that must stay inside the tolerance:
    assert expected + 7 == pytest.approx(expected, rel=0.10)
    assert gateway_module._count_text_tokens(MODELS["openai"], PROMPT) == expected


def test_unknown_models_fall_back_to_o200k(real_encoding):
    assert gateway_module._encoding_for("gpt-not-a-real-model").name == "o200k_base"