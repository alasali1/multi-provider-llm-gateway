"""Key handling and secret hygiene. Offline: no network, no real keys."""

from __future__ import annotations

import logging
import traceback

import pytest

import gateway as gateway_module
from fake_providers import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    build,
    first_token,
    make_gateway,
    run_stream,
    sdk_connection_error,
    sdk_status_error,
)
from gateway import UnifiedLLMGateway
from schemas import AuthenticationError, GatewayError, GatewayProviderError

PROVIDERS = pytest.mark.parametrize("provider", ["openai", "anthropic"])
ENV_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
MODELS = {"openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL}

# Realistic-looking but fake keys.
SECRETS = {
    "openai": "sk-proj-Zx9Qm3VbT7nKd2Lw8RpY4HcJ6FsA1GeU5oBiN0XtC",
    "anthropic": "sk-ant-api03-Qw3Er5Ty7Ui9Op1As2Df4Gh6Jk8Lz0XcVbNm3Qw5Er7Ty9Ui1Op",
}


# --------------------------------------------------------------------------- #
# Test 4: missing or malformed keys -> descriptive AuthenticationError
# --------------------------------------------------------------------------- #
@PROVIDERS
def test_missing_key_raises_descriptive_authentication_error(provider):
    gw = UnifiedLLMGateway()  # constructing with no keys at all must not crash

    with pytest.raises(AuthenticationError) as info:
        run_stream(gw, "hi", MODELS[provider])

    err = info.value
    assert isinstance(err, GatewayProviderError)
    assert err.provider == provider
    assert "No API key configured" in str(err)
    assert ENV_VARS[provider] in str(err), "message should name the variable to set"


MALFORMED_KEYS = {
    "empty": "",
    "whitespace_only": "   ",
    "inner_space": "sk-abc def",
    "inner_tab": "sk-abc\tdef",
    "inner_newline": "sk-abc\ndef",
    "double_quoted": '"sk-abcdef123456"',
    "single_quoted": "'sk-abcdef123456'",
    "non_ascii": "sk-clé-abcdef123456",
    "zero_width_char": "sk-abc​def123456",
}


@PROVIDERS
@pytest.mark.parametrize("value", list(MALFORMED_KEYS.values()), ids=list(MALFORMED_KEYS))
def test_malformed_key_raises_descriptive_authentication_error(provider, value, monkeypatch):
    monkeypatch.setenv(ENV_VARS[provider], value)
    gw = UnifiedLLMGateway()  # must not crash at construction

    with pytest.raises(AuthenticationError) as info:
        run_stream(gw, "hi", MODELS[provider])

    message = str(info.value)
    assert ENV_VARS[provider] in message
    assert ("No API key configured" if not value.strip() else "malformed") in message
    if value.strip():
        assert value.strip() not in message, "a rejected key must not be echoed back"


def test_key_with_surrounding_whitespace_is_accepted():
    # Typical .env / copy-paste artefact: harmless, so it is cleaned rather than rejected.
    gw = UnifiedLLMGateway(openai_api_key="sk-test-abc123456789\n", anthropic_api_key="  sk-ant-test-abc123456789 ")
    assert gw._key_errors == {}


def test_one_bad_key_does_not_break_the_other_provider():
    gw, model, _ = build("anthropic", openai_key="not a valid key")  # openai key is malformed
    assert run_stream(gw, "hi", model)[-1].is_final  # anthropic still works
    with pytest.raises(AuthenticationError):
        run_stream(gw, "hi", OPENAI_MODEL)


@PROVIDERS
@pytest.mark.parametrize("status,class_name", [(401, "AuthenticationError"), (403, "PermissionDeniedError")])
def test_provider_rejecting_the_key_becomes_authentication_error(provider, status, class_name):
    error = sdk_status_error(provider, class_name, status, "Incorrect API key provided")
    gw, model, _ = build(provider, create_error=error)
    with pytest.raises(AuthenticationError) as info:
        run_stream(gw, "hi", model)
    assert info.value.status_code == status
    assert info.value.provider == provider


@PROVIDERS
@pytest.mark.parametrize(
    "exc",
    [RuntimeError("boom"), KeyError("k"), ConnectionResetError("reset"), TimeoutError(), ValueError("bad")],
    ids=lambda e: type(e).__name__,
)
def test_unexpected_errors_never_escape_raw(provider, exc):
    gw, model, _ = build(provider, items=[first_token(provider), exc])
    with pytest.raises(GatewayProviderError) as info:
        run_stream(gw, "hi", model)
    assert not isinstance(info.value, type(exc))


# --------------------------------------------------------------------------- #
# Test 5: secret redaction
# --------------------------------------------------------------------------- #
def assert_no_secret_fragment(label: str, text: str, window: int = 12) -> None:
    """Fail if any 12-character slice of either fake key appears in `text`."""
    for secret in SECRETS.values():
        leaked = [
            secret[i : i + window]
            for i in range(len(secret) - window + 1)
            if secret[i : i + window] in text
        ]
        assert not leaked, f"{label} leaks part of an API key"


LEAKY_ERRORS = {
    "auth_401": lambda p, s: sdk_status_error(
        p, "AuthenticationError", 401, f"Incorrect API key provided: {s}",
        body={"error": {"message": f"bad key {s}"}},
    ),
    "rate_limit_429": lambda p, s: sdk_status_error(
        p, "RateLimitError", 429, f"Rate limited for key {s}", headers={"retry-after": "1"}
    ),
    "bad_request_400": lambda p, s: sdk_status_error(p, "BadRequestError", 400, f"invalid request for {s}"),
    "connection": lambda p, s: sdk_connection_error(p, f"proxy rejected Authorization: Bearer {s}"),
    "unexpected_runtime_error": lambda p, s: RuntimeError(f"boom while using {s}"),
}


@PROVIDERS
@pytest.mark.parametrize("where", ["on_create", "mid_stream"])
@pytest.mark.parametrize("case", list(LEAKY_ERRORS))
def test_errors_and_logs_never_contain_the_api_key(provider, where, case, caplog, capsys):
    caplog.set_level(logging.DEBUG)  # capture every logger at the most verbose level
    error = LEAKY_ERRORS[case](provider, SECRETS[provider])
    keys = {"openai_key": SECRETS["openai"], "anthropic_key": SECRETS["anthropic"]}
    if where == "on_create":
        gw, model, _ = build(provider, create_error=error, **keys)
    else:
        gw, model, _ = build(provider, items=[first_token(provider), error], **keys)

    with pytest.raises(GatewayError) as info:
        run_stream(gw, "hello", model)

    exc = info.value
    out, err_out = capsys.readouterr()
    assert isinstance(exc, GatewayProviderError)
    artefacts = {
        "str(exception)": str(exc),
        "repr(exception)": repr(exc),
        "exception args": repr(exc.args),
        "exception attributes": repr(vars(exc)),
        "formatted traceback": "".join(traceback.format_exception(exc)),
        "log output": caplog.text,
        "stdout": out,
        "stderr": err_out,
        "repr(gateway)": repr(gw),
        "str(gateway)": str(gw),
    }
    for label, text in artefacts.items():
        assert_no_secret_fragment(label, text)


def test_successful_stream_logs_nothing_sensitive(caplog):
    caplog.set_level(logging.DEBUG)
    keys = {"openai_key": SECRETS["openai"], "anthropic_key": SECRETS["anthropic"]}
    for provider in ("openai", "anthropic"):
        gw, model, _ = build(provider, **keys)
        run_stream(gw, "hello", model)
    assert_no_secret_fragment("log output", caplog.text)


def test_gateway_repr_never_shows_keys():
    gw = make_gateway(openai_key=SECRETS["openai"], anthropic_key=SECRETS["anthropic"])
    assert_no_secret_fragment("repr(gateway)", repr(gw))
    assert_no_secret_fragment("str(gateway)", str(gw))


@pytest.mark.parametrize(
    "text",
    [
        f"Incorrect API key provided: {SECRETS['openai']}",
        f"Authorization: Bearer {SECRETS['openai']}",
        f"x-api-key: {SECRETS['anthropic']}",
        f"key={SECRETS['anthropic']}&other=1",
    ],
    ids=["openai_message", "bearer_header", "anthropic_header", "query_string"],
)
def test_redact_scrubs_key_shaped_tokens(text):
    cleaned = gateway_module._redact(text)
    assert "[REDACTED]" in cleaned
    assert_no_secret_fragment("redacted text", cleaned)
