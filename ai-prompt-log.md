# Step 3: Human-in-the-Loop Code Audit

## Files Reviewed
- `schemas.py`
- `pricing.py`
- `gateway.py`

Scope: the four Step 3 checklist items (event-loop blocking, streaming delta parsing, telemetry/timing and cost rounding, client instantiation), plus the secret-leak requirement and the problems we hit while running the code. Findings from the earlier automated review that did not bear on these were dropped (see the note at the end).

---

## Finding 1

**File:** `gateway.py`
**Location:** `_openai_deltas()` and `stream()` (timer start)
**Category:** Bug
**Security:** None

**What I found:** The latency/TTFT clock (`_RunState(t0=...)`) started before the local tiktoken prompt estimate, which runs before the OpenAI request. TTFT and total latency therefore included our own tokenizing time, and on a cold start tiktoken's one-time encoding download.

**Why it matters:** Reported TTFT and tokens/sec were inflated by local overhead, not provider latency (the "timing drift" in the checklist). The Anthropic path has no local pre-request work, so it was already correct.

**Recommended fix:** Reset `state.t0 = time.perf_counter()` immediately before `client.chat.completions.create(...)`.

**Status:** Fixed. Regression test: `test_openai_timing_excludes_local_tokenizer_work`.

---

## Finding 2

**File:** `gateway.py`
**Location:** `_anthropic_deltas()` (`messages.create` call)
**Category:** Bug
**Security:** None

**What I found:** The first draft passed `temperature=` to Anthropic. `anthropic>=1.0` removed that keyword (TypeError), and after working around it, the live API returned HTTP 400 because Claude models released after Opus 4.6 (including the 5.5 family) reject any value other than 1.0.

**Why it matters:** Every Anthropic request failed. Fake-client tests could not catch it.

**Recommended fix:** Do not send temperature to Anthropic; keep the range check in `stream()` for a uniform interface.

**Status:** Fixed. Tests assert neither `temperature` nor `extra_body` is sent.

---

## Finding 3

**File:** `pricing.py`, `app.py`
**Location:** `PRICING`, `MODEL_CATALOG`
**Category:** Bug
**Security:** None

**What I found:** Retired model names (`claude-3-5-sonnet`, `claude-3-5-haiku`) were hard-coded, producing HTTP 404 `NotFoundError`. Rates are also a hard-coded snapshot with no source or effective date.

**Why it matters:** Selecting Anthropic failed, and costs can drift from real prices.

**Recommended fix:** Use current models (`claude-sonnet-5-5`, `claude-haiku-5-5`) and re-verify rates against provider pricing pages.

**Status:** Fixed (names). Pricing source/date remains a documented limitation; the Haiku rate is approximate.

---

## Finding 4

**File:** `gateway.py`
**Location:** `_anthropic_deltas()` (`message_start` / `message_delta` handling) and `_build_telemetry()`
**Category:** Bug
**Security:** None

**What I found:** `message_start` sets `state.output_tokens` to a small initial value. If the stream ends before `message_delta` arrives, that value is not `None`, so `_build_telemetry` labels it `usage_source="provider"` instead of estimating. Verified by reading the code.

**Why it matters:** A truncated stream could report a too-low output count and cost as if it were final provider usage.

**Recommended fix:** Track whether final usage was received (for example, set `output_tokens` only from `message_delta`, keeping the `message_start` value separate), and fall back to the estimate otherwise.

**Status:** Confirmed by inspection; not fixed and no test yet. Narrow case: normal streams always send `message_delta`.

---

## Finding 5

**File:** `gateway.py`
**Location:** `_anthropic_deltas()` (`message_stop`) and `stream()` (early termination)
**Category:** Missing requirement
**Security:** None

**What I found:** `message_stop` is not handled explicitly; it falls through and is ignored, which is correct because the loop ends with the stream. The stream is wrapped in `async with stream:`, so it closes if the consumer stops reading. The checklist's `content_block_delta` / `message_delta` / `message_stop` distinctions are covered by `an_*` event builders in the fakes and a real-SDK round trip.

**Why it matters:** Confirms the checklist's Anthropic event handling does not crash or double count.

**Recommended fix:** Add an explicit test for consumer cancellation / early `aclose()`.

**Status:** Partly verified (event handling tested); cancellation not tested.

---

## Finding 6

**File:** `gateway.py`
**Location:** `stream()` exception handler
**Category:** Bug
**Security:** None

**What I found:** `_build_telemetry` runs inside the same `try` as the provider stream, with a broad `except Exception` that calls `_translate_error(exc, provider)`. A local failure (tokenizer, cost calculation) would be reported as a provider error. Verified by reading the code.

**Why it matters:** Misleading error messages and harder debugging. Text has already been yielded by that point, so the consumer gets no final telemetry chunk.

**Recommended fix:** Move telemetry construction outside the provider `try`, and handle its failures separately.

**Status:** Confirmed by inspection; not fixed.

---

## Finding 7

**File:** `pricing.py`
**Location:** `compute_cost()` and `resolve_pricing()`
**Category:** Anti-pattern
**Security:** None

**What I found:** Cost uses floats rounded to 10 decimals (`round(x, 10)`) rather than the assignment's suggested `round(cost, 6)`; the UI displays six decimals. Separately, `resolve_pricing` uses longest-prefix matching, so an unsupported ID starting with a supported name would receive that model's price. In the app, `MODEL_CATALOG` limits which models can be requested.

**Why it matters:** The rounding deviates from the stated guidance (stored value is more precise than displayed). The prefix match is a minor risk for direct library callers.

**Recommended fix:** Decide between 10 and 6 decimals (changing it requires updating tests that use `pytest.approx`); use an explicit model list if exact IDs are required.

**Status:** Confirmed; kept by design for now, decision pending.

---

## Finding 8

**File:** `gateway.py`, `pricing.py`, `Dockerfile`
**Location:** `from schemas import ...`, `from pricing import ...`
**Category:** Bug
**Security:** None

**What I found:** Flat top-level imports depend on the launch layout. In Docker the container crash-looped with `ModuleNotFoundError: No module named 'gateway'` because only `src/` was copied.

**Why it matters:** The app fails to start if files are not where the imports expect.

**Recommended fix:** One consistent flat layout, `PYTHONPATH=/app` and `COPY . ./` in the Dockerfile, `pythonpath = .` in `pytest.ini`.

**Status:** Fixed. Container reports healthy.

---

## Finding 9

**File:** `schemas.py`, `gateway.py`
**Location:** `GatewayProviderError.__str__()`, `_translate_error()`, `_redact()`
**Category:** Security risk (reviewed)
**Security:** Low

**What I found:** `__str__` prints `args[0]`, but `_translate_error` only passes fixed messages ("Rate limit exceeded", "Invalid or unauthorized API key", and so on) plus status and request id, never the raw SDK text. `_redact` is used once, on the exception class name in the final fallback, so it is nearly redundant: the earlier claim that it is unused was not accurate.

**Why it matters:** No path found that forwards a key or raw provider message to the UI or logs. Logging uses no `exc_info`. `test_security.py` checks this.

**Recommended fix:** Keep forwarding only fixed messages; optionally remove `_redact` or apply it consistently.

**Status:** No exploitable issue found. Redundant helper noted.

---

## Checklist items with no bug found

- **Event-loop blocking:** only `AsyncOpenAI` / `AsyncAnthropic`; no `time.sleep` or blocking I/O; tiktoken work runs in `asyncio.to_thread`.
- **OpenAI delta parsing:** `if chunk.usage is not None`, `if chunk.choices`, and `if text` guard the empty-choices usage chunk and `delta.content is None`.
- **Client instantiation:** both clients created once in `UnifiedLLMGateway.__init__`, reused, closed with `aclose()`; one gateway shared via FastAPI `lifespan`.

---

# Review Summary

- **Bugs identified:** 6 (Findings 1, 2, 3, 4, 6, 8)
- **Security risks reviewed:** 1 (Finding 9, none exploitable)
- **Anti-patterns identified:** 1 (Finding 7)
- **Missing requirements identified:** 1 (Finding 5)
- **Total findings:** 9
- **Fixed:** 1, 2, 3 (names), 8. **Open:** 4, 6 (confirmed by inspection), 5 (cancellation test), 7 (decision pending).

**Audit note:** Findings 1, 2, 3 and 8 were reproduced by running the code or tests. Findings 4 and 6 are from source inspection and have not been reproduced at runtime. Test status: 129 passed, 1 skipped (opt-in live test). Dropped as low relevance: nonnegative-constraint validation, Decimal for money, config input validation, tiktoken fallback labelling.