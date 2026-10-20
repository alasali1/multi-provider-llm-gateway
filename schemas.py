"""Pydantic schemas and domain exceptions for the unified LLM gateway."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ChunkTelemetry(BaseModel):
    """Usage, latency and cost metrics. Attached to the final chunk only."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0, description="Prompt tokens billed by the provider.")
    output_tokens: int = Field(ge=0, description="Completion tokens billed by the provider.")
    total_latency_s: float = Field(ge=0, description="Request start to stream end, in seconds.")
    ttft_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to first text token, in seconds (None if no text was produced).",
    )
    tokens_per_second: float = Field(
        ge=0,
        description=(
            "Decode throughput: output_tokens / (total_latency - ttft). "
            "Falls back to total latency when TTFT is unavailable."
        ),
    )
    input_cost_usd: float = Field(ge=0)
    output_cost_usd: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    usage_source: Literal["provider", "estimated"] = Field(
        default="provider",
        description="'estimated' means the provider returned no usage and tiktoken was used.",
    )
    local_prompt_tokens_estimate: Optional[int] = Field(
        default=None,
        description="tiktoken prompt-token estimate (OpenAI only) used to cross-check usage.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds (derived from `ttft_s`)."""
        return None if self.ttft_s is None else round(self.ttft_s * 1000, 3)


class StreamChunk(BaseModel):
    """Provider-agnostic streaming event."""

    model_config = ConfigDict(frozen=True)

    delta_text: str
    provider: str
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class GatewayError(Exception):
    """Base class for every error raised by the gateway."""


class UnsupportedModelError(GatewayError):
    """The requested model is not in the pricing/provider registry."""


class GatewayProviderError(GatewayError):
    """A provider call failed. Messages are sanitized and never contain secrets."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.request_id = request_id

    def __str__(self) -> str:  # stable, secret-free representation
        parts = [f"[{self.provider}] {self.args[0]}"]
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " ".join(parts)


class RateLimitExceededError(GatewayProviderError):
    """HTTP 429 / provider rate limit. `retry_after_s` is set when the provider sends it."""

    def __init__(self, message: str, *, retry_after_s: Optional[float] = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.retry_after_s = retry_after_s


class AuthenticationFailedError(GatewayProviderError):
    """Missing, malformed, invalid or unauthorized API key."""


# Short alias: `except AuthenticationError` reads naturally in calling code.
AuthenticationError = AuthenticationFailedError


class ProviderConnectionError(GatewayProviderError):
    """Network failure or timeout while talking to the provider."""
