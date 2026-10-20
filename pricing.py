"""Static pricing registry and cost calculation.

Rates are USD per 1,000,000 tokens. They are a point-in-time snapshot: verify
against each provider's pricing page and update before relying on cost numbers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas import UnsupportedModelError

Provider = Literal["openai", "anthropic"]


class ModelPricing(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    input_per_mtok: float = Field(ge=0, description="USD per 1M input tokens")
    output_per_mtok: float = Field(ge=0, description="USD per 1M output tokens")


class CostBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


# Keys are model-family prefixes; dated snapshots (e.g. "gpt-4o-2024-08-06",
# "claude-sonnet-5-5") resolve via longest-prefix match.
PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o-mini": ModelPricing(provider="openai", input_per_mtok=0.15, output_per_mtok=0.60),
    "gpt-4o": ModelPricing(provider="openai", input_per_mtok=2.50, output_per_mtok=10.00),
    # Anthropic
    "claude-sonnet-5-5": ModelPricing(provider="anthropic", input_per_mtok=2.00, output_per_mtok=10.00),
    "claude-haiku-5-5": ModelPricing(provider="anthropic", input_per_mtok=0.10, output_per_mtok=0.50),
}


def resolve_pricing(model: str) -> ModelPricing:
    """Return pricing for `model` (exact key first, then longest matching prefix)."""
    if model in PRICING:
        return PRICING[model]
    matches = [key for key in PRICING if model.startswith(key)]
    if not matches:
        supported = ", ".join(sorted(PRICING))
        raise UnsupportedModelError(
            f"Unsupported model {model!r}. Supported model families: {supported}"
        )
    return PRICING[max(matches, key=len)]


def compute_cost(pricing: ModelPricing, input_tokens: int, output_tokens: int) -> CostBreakdown:
    input_cost = input_tokens * pricing.input_per_mtok / 1_000_000
    output_cost = output_tokens * pricing.output_per_mtok / 1_000_000
    return CostBreakdown(
        input_cost_usd=round(input_cost, 10),
        output_cost_usd=round(output_cost, 10),
        total_cost_usd=round(input_cost + output_cost, 10),
    )