"""UnifiedLLMGateway: one async streaming interface over OpenAI and Anthropic.

Requirements: python>=3.11, pydantic>=2, openai>=3.0, anthropic>=1.0, tiktoken
(the SDKs bring their own HTTP library, httpx2)

Example:
    gateway = UnifiedLLMGateway()  # reads OPENAI_API_KEY / ANTHROPIC_API_KEY
    async for chunk in gateway.stream("Say hi", model="gpt-4o-mini"):
        print(chunk.delta_text, end="", flush=True)
        if chunk.is_final:
            print(chunk.telemetry)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional

import anthropic
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from pricing import ModelPricing, compute_cost, resolve_pricing
from schemas import (
    AuthenticationFailedError,
    ChunkTelemetry,
    GatewayError,
    GatewayProviderError,
    ProviderConnectionError,
    RateLimitExceededError,
    StreamChunk,
)

logger = logging.getLogger(__name__)

_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]{4,}|Bearer\s+\S+|x-api-key\S*)", re.IGNORECASE)
_OPENAI_FALLBACK_ENCODING = "o200k_base"
# Per-message framing overhead used by OpenAI chat models (3) + reply priming (3).
_OPENAI_CHAT_OVERHEAD = 6
_PROMPT_MISMATCH_WARN_THRESHOLD = 8
_ENV_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def _key_problem(key: Optional[str]) -> Optional[str]:
    """Classify an API key as 'missing', 'malformed' or None (structurally usable).

    Only the key's shape is inspected, and the key itself is never echoed.
    Surrounding whitespace (e.g. a trailing newline from a .env file) is tolerated.
    """
    if key is None or not key.strip():
        return "missing"
    key = key.strip()
    if key[0] in "'\"" or key[-1] in "'\"":  # quotes pasted in from a .env / shell line
        return "malformed"
    if any(not (32 < ord(c) < 127) for c in key):  # spaces, control chars, non-ASCII
        return "malformed"
    return None


def _redact(text: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", text)


@functools.lru_cache(maxsize=16)
def _encoding_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_OPENAI_FALLBACK_ENCODING)


def _count_openai_prompt_tokens(model: str, prompt: str) -> int:
    return len(_encoding_for(model).encode(prompt)) + _OPENAI_CHAT_OVERHEAD


def _count_text_tokens(model: str, text: str) -> int:
    return len(_encoding_for(model).encode(text))


@dataclass
class _RunState:
    """Mutable metrics collected while a provider stream is consumed."""

    t0: float
    ttft_s: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    local_prompt_estimate: Optional[int] = None

    def mark_text(self) -> None:
        if self.ttft_s is None:
            self.ttft_s = time.perf_counter() - self.t0


class UnifiedLLMGateway:
    def __init__(
        self,
        *,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        default_max_tokens: int = 4096,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._default_max_tokens = default_max_tokens
        self._key_errors: dict[str, str] = {}
        openai_key = self._usable_key(
            "openai", openai_api_key or os.getenv(_ENV_VARS["openai"])
        )
        anthropic_key = self._usable_key(
            "anthropic", anthropic_api_key or os.getenv(_ENV_VARS["anthropic"])
        )
        self._openai: Optional[AsyncOpenAI] = (
            AsyncOpenAI(api_key=openai_key, timeout=timeout_s, max_retries=max_retries)
            if openai_key
            else None
        )
        self._anthropic: Optional[AsyncAnthropic] = (
            AsyncAnthropic(api_key=anthropic_key, timeout=timeout_s, max_retries=max_retries)
            if anthropic_key
            else None
        )
#presentation -explain 1
    def _usable_key(self, provider: str, raw: Optional[str]) -> Optional[str]:
        """Return the cleaned key, or record why it is unusable (raised lazily on first use)."""
        problem = _key_problem(raw)
        if problem:
            self._key_errors[provider] = problem
            return None
        return raw.strip()  # type: ignore[union-attr]

    def __repr__(self) -> str:  # never expose keys
        return (
            f"UnifiedLLMGateway(openai={'on' if self._openai else 'off'}, "
            f"anthropic={'on' if self._anthropic else 'off'})"
        )

    async def aclose(self) -> None:
        if self._openai:
            await self._openai.close()
        if self._anthropic:
            await self._anthropic.close()

    async def __aenter__(self) -> "UnifiedLLMGateway":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def stream(
        self, prompt: str, model: str, temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        pricing = resolve_pricing(model)  # raises UnsupportedModelError
        provider = pricing.provider
        if provider == "anthropic" and not 0.0 <= temperature <= 1.0:
            raise ValueError("Anthropic temperature must be between 0.0 and 1.0")
        if provider == "openai" and not 0.0 <= temperature <= 2.0:
            raise ValueError("OpenAI temperature must be between 0.0 and 2.0")

        state = _RunState(t0=time.perf_counter())
        text_parts: list[str] = []
        source = self._openai_deltas if provider == "openai" else self._anthropic_deltas

        try:
            async for delta in source(prompt, model, temperature, state):
                text_parts.append(delta)
                yield StreamChunk(delta_text=delta, provider=provider, model=model)
            latency = time.perf_counter() - state.t0
            telemetry = await self._build_telemetry(
                model, pricing, state, latency, "".join(text_parts)
            )
        except GatewayError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated to a domain error
            raise self._translate_error(exc, provider) from None

        yield StreamChunk(
            delta_text="", provider=provider, model=model, is_final=True, telemetry=telemetry
        )

    # ------------------------------------------------------------------ #
    # Provider adapters: yield plain text deltas, record metrics in state
    # ------------------------------------------------------------------ #
    async def _openai_deltas(
        self, prompt: str, model: str, temperature: float, state: _RunState
    ) -> AsyncIterator[str]:
        client = self._require(self._openai, "openai")
        state.local_prompt_estimate = await asyncio.to_thread(
            _count_openai_prompt_tokens, model, prompt
        )
         # Start the clock only now: local tokenizing (and tiktoken's one-time encoding download)
         # is our own overhead, not provider latency, and must not inflate TTFT or total latency.
        state.t0 = time.perf_counter()
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async with stream:
            async for chunk in stream:
                if chunk.usage is not None:  # final chunk: empty `choices`, populated usage
                    state.input_tokens = chunk.usage.prompt_tokens
                    state.output_tokens = chunk.usage.completion_tokens
                if chunk.choices:
                    text = chunk.choices[0].delta.content
                    if text:
                        state.mark_text()
                        yield text

    async def _anthropic_deltas(
        self, prompt: str, model: str, temperature: float, state: _RunState
    ) -> AsyncIterator[str]:
        client = self._require(self._anthropic, "anthropic")
        # Temperature is deliberately NOT sent: Claude models released after Opus 4.6 (incl. the
        # 5.5 family) reject any value other than 1.0 with HTTP 400, and anthropic>=1.0 removed the
        # keyword anyway. The argument is still range-checked in stream() for a uniform interface.
        stream = await client.messages.create(
            model=model,
            max_tokens=self._default_max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async with stream:
            async for event in stream:
                etype = event.type
                if etype == "message_start":
                    usage = event.message.usage
                    state.input_tokens = usage.input_tokens
                    state.output_tokens = usage.output_tokens  # small initial value
                elif etype == "content_block_delta":
                    if event.delta.type == "text_delta" and event.delta.text:
                        state.mark_text()
                        yield event.delta.text
                elif etype == "message_delta":
                    # `usage.output_tokens` is cumulative, so overwrite rather than add.
                    state.output_tokens = event.usage.output_tokens
                    input_tokens = getattr(event.usage, "input_tokens", None)
                    if input_tokens is not None:
                        state.input_tokens = input_tokens

    # ------------------------------------------------------------------ #
    # Telemetry
    # ------------------------------------------------------------------ #
    async def _build_telemetry(
        self,
        model: str,
        pricing: ModelPricing,
        state: _RunState,
        latency_s: float,
        full_text: str,
    ) -> ChunkTelemetry:
        usage_source = "provider"
        input_tokens = state.input_tokens
        output_tokens = state.output_tokens

        if input_tokens is None or output_tokens is None:
            # Provider sent no usage (e.g. stream interrupted); fall back to tiktoken.
            usage_source = "estimated"
            if input_tokens is None:
                input_tokens = state.local_prompt_estimate or 0
            if output_tokens is None:
                output_tokens = await asyncio.to_thread(_count_text_tokens, model, full_text)

        if (
            pricing.provider == "openai"
            and usage_source == "provider"
            and state.local_prompt_estimate is not None
            and abs(state.local_prompt_estimate - input_tokens) > _PROMPT_MISMATCH_WARN_THRESHOLD
        ):
            logger.warning(
                "Prompt token mismatch for %s: provider=%d local_estimate=%d",
                model,
                input_tokens,
                state.local_prompt_estimate,
            )

        ttft = state.ttft_s
        gen_time = latency_s - ttft if ttft is not None else latency_s
        tps = output_tokens / gen_time if gen_time > 0 else 0.0
        cost = compute_cost(pricing, input_tokens, output_tokens)

        return ChunkTelemetry(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_latency_s=round(latency_s, 6),
            ttft_s=round(ttft, 6) if ttft is not None else None,
            tokens_per_second=round(tps, 3),
            input_cost_usd=cost.input_cost_usd,
            output_cost_usd=cost.output_cost_usd,
            total_cost_usd=cost.total_cost_usd,
            usage_source=usage_source,
            local_prompt_tokens_estimate=state.local_prompt_estimate,
        )

    # ------------------------------------------------------------------ #
    # Errors
    # ------------------------------------------------------------------ #
    def _require(self, client: Any, provider: str) -> Any:
        if client is not None:
            return client
        env_var = _ENV_VARS[provider]
        if self._key_errors.get(provider) == "malformed":
            raise AuthenticationFailedError(
                f"API key for {provider} is malformed: it must be a single token with no "
                f"spaces, quotes or line breaks. Check the {env_var} environment variable.",
                provider=provider,
            )
        raise AuthenticationFailedError(
            f"No API key configured for {provider}. Set the {env_var} environment "
            "variable (or pass the key to UnifiedLLMGateway).",
            provider=provider,
        )

    @staticmethod
    def _translate_error(exc: Exception, provider: str) -> GatewayError:
        """Map SDK exceptions to domain errors. Raw SDK messages are never forwarded
        (they can echo key fragments); only status code and request id are kept."""
        sdk = openai if provider == "openai" else anthropic
        status = getattr(exc, "status_code", None)
        request_id = getattr(exc, "request_id", None)
        common = {"provider": provider, "status_code": status, "request_id": request_id}

        logger.error(
            "%s call failed: %s (status=%s request_id=%s)",
            provider,
            type(exc).__name__,
            status,
            request_id,
        )  # no exc_info: avoids stack traces and headers in logs

        if isinstance(exc, sdk.RateLimitError):
            retry_after: Optional[float] = None
            response = getattr(exc, "response", None)
            raw = response.headers.get("retry-after") if response is not None else None
            try:
                retry_after = float(raw) if raw is not None else None
            except ValueError:
                retry_after = None
            return RateLimitExceededError("Rate limit exceeded", retry_after_s=retry_after, **common)
        if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
            return AuthenticationFailedError("Invalid or unauthorized API key", **common)
        if isinstance(exc, sdk.APITimeoutError):
            return ProviderConnectionError("Request to provider timed out", **common)
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderConnectionError("Could not connect to provider", **common)
        if isinstance(exc, sdk.APIStatusError):
            return GatewayProviderError(f"Provider returned an error ({type(exc).__name__})", **common)
        return GatewayProviderError(
            f"Unexpected failure: {_redact(type(exc).__name__)}", **common
        )