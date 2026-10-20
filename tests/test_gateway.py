"""Gateway behaviour: stream protocol, token accounting, telemetry, error mapping.

Everything runs against fake provider clients (see fake_providers.py), so no
network access, API keys or money are needed. The one exception is the opt-in
`live` test at the bottom of the "Test 2" section.
"""

from __future__ import annotations

import os

import pytest
import tiktoken

import gateway as gateway_module
from fake_providers import (
    ANTHROPIC_MODEL,
    DEFAULT_INPUT_TOKENS,
    DEFAULT_OUTPUT_TOKENS,
    EXPECTED_TEXT,
    OPENAI_MODEL,
    TOKENS,
    FakeOpenAI,
    build,
    first_token,
    make_gateway,
    oa_finish,
    oa_role,
    oa_text,
    openai_script,
    run_stream,
    sdk_connection_error,
    sdk_status_error,
)
from gateway import UnifiedLLMGateway
from pricing import PRICING, compute_cost, resolve_pricing
from schemas import (
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
    StreamChunk,
    UnsupportedModelError,
)

PROVIDERS = pytest.mark.parametrize("provider", ["openai", "anthropic"])


@pytest.fixture(autouse=True)
def _stub_tokenizer(request, monkeypatch):
    """Keep tests off the tiktoken download unless a test opts in with `real_tokenizer`."""
    if request.node.get_closest_marker("real_tokenizer"):
        return
    monkeypatch.setattr(
        gateway_module, "_count_openai_prompt_tokens", lambda model, prompt: DEFAULT_INPUT_TOKENS
    )


# --------------------------------------------------------------------------- #
# Test 1: unified stream protocol
# --------------------------------------------------------------------------- #
@PROVIDERS
def test_stream_yields_valid_chunks_and_expected_text(provider):
    gw, model, _ = build(provider)
    chunks = run_stream(gw, "Say hello", model)

    assert chunks, "stream produced no chunks"
    assert all(isinstance(c, StreamChunk) for c in chunks)
    for chunk in chunks:  # structurally valid: survives a pydantic round-trip
        StreamChunk.model_validate(chunk.model_dump())
    assert {c.provider for c in chunks} == {provider}
    assert {c.model for c in chunks} == {model}

    *body, final = chunks
    assert final.is_final and final.delta_text == ""
    assert all(not c.is_final and c.telemetry is None for c in body)
    assert all(c.delta_text for c in body), "empty text deltas leaked to the consumer"
    # Provider-specific noise (role chunk, finish chunk, thinking/ping events) is hidden:
    assert [c.delta_text for c in body] == TOKENS
    assert "".join(c.delta_text for c in chunks) == EXPECTED_TEXT


def test_openai_request_enables_streaming_usage():
    gw, model, client = build("openai")
    run_stream(gw, "hi", model, temperature=0.2)
    (call,) = client.calls
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}
    assert call["temperature"] == 0.2
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_request_is_streaming_with_max_tokens():
    gw, model, client = build("anthropic")
    run_stream(gw, "hi", model, temperature=0.2)
    (call,) = client.calls
    assert call["stream"] is True
    # Claude 5.x rejects non-default temperature (HTTP 400), so it must never be sent.
    assert "temperature" not in call and "extra_body" not in call
    assert isinstance(call["max_tokens"], int) and call["max_tokens"] > 0
    assert call["messages"] == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------- #
# Test 2: token count validation (50 deterministic words vs tiktoken, +/-10%)
# --------------------------------------------------------------------------- #
WORDS = (
    "counterrevolutionary electroencephalography uncharacteristically internationalization "
    "disproportionately incomprehensibility thermodynamically institutionalization "
    "misinterpretation unconstitutionality overcompensation interdisciplinary "
    "photosynthesizing deindustrialization anthropomorphization counterintuitively "
    "immunohistochemistry microelectronics neuropsychological telecommunications "
    "crystallographically intergovernmental reconceptualization pharmacokinetics "
    "hypercholesterolemia biodegradability electromagnetically characteristically "
    "unsubstantiated misunderstandings decentralization acknowledgements transcontinental "
    "neurotransmitters gastrointestinal paleontological superconductivity interoperability "
    "philosophically comprehensiveness environmentalists reproducibility archaeologically "
    "cryptographically recontextualization microarchitecture multidimensional "
    "unpredictability hydrodynamically sociolinguistics"
).split()
PROMPT = " ".join(WORDS)
OPENAI_CHAT_FRAMING_TOKENS = 7  # per-request overhead OpenAI adds to chat prompts


def test_prompt_fixture_is_exactly_50_distinct_words():
    assert len(PROMPT.split()) == 50
    assert len(set(WORDS)) == 50


@pytest.fixture(scope="module")
def encoding():
    try:
        return tiktoken.encoding_for_model(OPENAI_MODEL)
    except Exception as exc:  # first use needs a one-time download of the encoding file
        pytest.skip(f"tiktoken encoding unavailable ({type(exc).__name__}); connect once to cache it")


@pytest.mark.real_tokenizer
def test_prompt_tokens_match_tiktoken_within_10_percent(encoding):
    expected = len(encoding.encode(PROMPT))
    # Simulated provider usage = tiktoken count + OpenAI's chat framing overhead.
    script = openai_script(TOKENS, expected + OPENAI_CHAT_FRAMING_TOKENS, 5)
    gw = make_gateway(openai_client=FakeOpenAI(script))

    telemetry = run_stream(gw, PROMPT, OPENAI_MODEL)[-1].telemetry

    assert telemetry.input_tokens == pytest.approx(expected, rel=0.10)
    # The gateway's own local tiktoken cross-check must agree as well:
    assert telemetry.local_prompt_tokens_estimate == pytest.approx(expected, rel=0.10)


@pytest.mark.live
@pytest.mark.real_tokenizer
@pytest.mark.skipif(
    not (os.getenv("RUN_LIVE_TESTS") == "1" and os.getenv("OPENAI_API_KEY")),
    reason="opt-in: set RUN_LIVE_TESTS=1 and OPENAI_API_KEY (costs a fraction of a cent)",
)
def test_live_openai_prompt_tokens_match_tiktoken_within_10_percent(encoding):
    expected = len(encoding.encode(PROMPT))
    gw = UnifiedLLMGateway()
    telemetry = run_stream(gw, PROMPT, OPENAI_MODEL)[-1].telemetry
    assert telemetry.usage_source == "provider"
    assert telemetry.input_tokens == pytest.approx(expected, rel=0.10)


# --------------------------------------------------------------------------- #
# Test 3: telemetry sanity on the final chunk
# --------------------------------------------------------------------------- #
def test_openai_timing_excludes_local_tokenizer_work(monkeypatch):
    """TTFT and latency measure the provider, not our own tiktoken estimate (which can be slow
    on a cold start because tiktoken downloads its encoding file once)."""
    import time

    def slow_tokenizer(model, prompt):
        time.sleep(0.3)
        return DEFAULT_INPUT_TOKENS

    monkeypatch.setattr(gateway_module, "_count_openai_prompt_tokens", slow_tokenizer)
    gw, model, _ = build("openai")
    t = run_stream(gw, "hi", model)[-1].telemetry

    assert t.ttft_s < 0.2, f"TTFT includes the local tokenizer delay: {t.ttft_s}s"
    assert t.total_latency_s < 0.2


@PROVIDERS
def test_final_chunk_telemetry_is_sane(provider):
    # delay=10ms per event gives real, strictly positive timings without slowing the suite.
    gw, model, _ = build(provider, delay=0.01)
    final = run_stream(gw, "hi", model)[-1]

    assert final.is_final
    t = final.telemetry
    assert t is not None
    assert t.total_cost_usd > 0
    assert t.tokens_per_second > 0
    assert t.ttft_ms > 0

    # Internal consistency
    assert t.ttft_ms == pytest.approx(t.ttft_s * 1000, abs=0.01)
    assert 0 < t.ttft_s <= t.total_latency_s
    # Exact counts from the provider (Anthropic's cumulative output must not be double counted):
    assert (t.input_tokens, t.output_tokens) == (DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS)
    assert t.usage_source == "provider"
    price = resolve_pricing(model)
    expected_cost = (
        DEFAULT_INPUT_TOKENS * price.input_per_mtok + DEFAULT_OUTPUT_TOKENS * price.output_per_mtok
    ) / 1_000_000
    assert t.total_cost_usd == pytest.approx(expected_cost)
    assert t.total_cost_usd == pytest.approx(t.input_cost_usd + t.output_cost_usd)


def test_cost_is_exact_to_six_decimals():
    cost = compute_cost(resolve_pricing("gpt-4o-mini"), 800, 250)
    assert f"{cost.total_cost_usd:.6f}" == "0.000270"


@pytest.mark.parametrize("model", sorted(PRICING))
def test_pricing_table_drives_cost_math(model):
    price = PRICING[model]
    cost = compute_cost(price, 1_000_000, 1_000_000)
    assert cost.input_cost_usd == pytest.approx(price.input_per_mtok)
    assert cost.output_cost_usd == pytest.approx(price.output_per_mtok)


def test_dated_model_snapshots_resolve_to_the_most_specific_family():
    # gpt-4o-mini must never be billed at gpt-4o rates by a prefix collision.
    assert resolve_pricing("gpt-4o-mini-2024-07-18") is PRICING["gpt-4o-mini"]
    assert resolve_pricing("claude-haiku-5-5") is PRICING["claude-haiku-5-5"]


def test_missing_usage_falls_back_to_estimate(monkeypatch):
    monkeypatch.setattr(gateway_module, "_count_text_tokens", lambda model, text: len(text))
    client = FakeOpenAI([oa_role(), oa_text("abc"), oa_text("de"), oa_finish()])  # no usage chunk
    t = run_stream(make_gateway(openai_client=client), "hi", OPENAI_MODEL)[-1].telemetry
    assert t.usage_source == "estimated"
    assert t.output_tokens == 5


# --------------------------------------------------------------------------- #
# Reliability extras: SDK errors become typed domain errors
# --------------------------------------------------------------------------- #
ERROR_CASES = {
    "rate_limit": (
        lambda p: sdk_status_error(p, "RateLimitError", 429, "slow down", headers={"retry-after": "2"}),
        RateLimitExceededError,
    ),
    "connection": (lambda p: sdk_connection_error(p), ProviderConnectionError),
    "timeout": (lambda p: sdk_connection_error(p, timeout=True), ProviderConnectionError),
    "server_error": (lambda p: sdk_status_error(p, "InternalServerError", 500, "oops"), GatewayProviderError),
}


@PROVIDERS
@pytest.mark.parametrize("where", ["on_create", "mid_stream"])
@pytest.mark.parametrize("case", list(ERROR_CASES))
def test_sdk_errors_become_domain_errors(provider, where, case):
    make_error, expected = ERROR_CASES[case]
    error = make_error(provider)
    if where == "on_create":
        gw, model, _ = build(provider, create_error=error)
    else:
        gw, model, _ = build(provider, items=[first_token(provider), error])

    with pytest.raises(GatewayProviderError) as info:
        run_stream(gw, "hi", model)
    assert type(info.value) is expected
    assert info.value.provider == provider


@PROVIDERS
def test_rate_limit_exposes_retry_after_and_status(provider):
    error = sdk_status_error(provider, "RateLimitError", 429, "slow down", headers={"retry-after": "2"})
    gw, model, _ = build(provider, create_error=error)
    with pytest.raises(RateLimitExceededError) as info:
        run_stream(gw, "hi", model)
    assert info.value.retry_after_s == 2.0
    assert info.value.status_code == 429


def test_unsupported_model_is_rejected_before_any_request():
    gw, _, client = build("openai")
    with pytest.raises(UnsupportedModelError):
        run_stream(gw, "hi", "not-a-model")
    assert client.calls == []


def test_anthropic_rejects_out_of_range_temperature():
    gw, model, client = build("anthropic")
    with pytest.raises(ValueError):
        run_stream(gw, "hi", model, temperature=1.5)
    assert client.calls == []