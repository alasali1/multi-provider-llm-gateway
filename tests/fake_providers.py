"""Offline stand-ins for the OpenAI / Anthropic async clients.

They mimic only the attributes UnifiedLLMGateway actually reads, so the suite
needs no network, no API keys, and costs nothing to run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any, Optional

import anthropic
import openai

from gateway import UnifiedLLMGateway
from http_lib import http_module
from schemas import StreamChunk

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-haiku-5-5"
FAKE_OPENAI_KEY = "sk-test-" + "A" * 24
FAKE_ANTHROPIC_KEY = "sk-ant-test-" + "B" * 24

TOKENS = ["Hel", "lo", ", ", "wor", "ld", "!"]
EXPECTED_TEXT = "".join(TOKENS)  # "Hello, world!"
DEFAULT_INPUT_TOKENS = 120
DEFAULT_OUTPUT_TOKENS = 40


# --------------------------------------------------------------------------- #
# Fake streaming clients
# --------------------------------------------------------------------------- #
class FakeStream:
    """Async-context-manager + async-iterator, like the SDKs' stream objects."""

    def __init__(self, items: Iterable[Any], delay: float = 0.0) -> None:
        self._items = list(items)
        self._delay = delay

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self._items:
            if self._delay:
                await asyncio.sleep(self._delay)
            if isinstance(item, BaseException):
                raise item  # simulates a failure part-way through the stream
            yield item


class _Endpoint:
    def __init__(self, owner: "_FakeClient") -> None:
        self._owner = owner

    async def create(self, **kwargs: Any) -> FakeStream:
        self._owner.calls.append(kwargs)
        if self._owner.create_error is not None:
            raise self._owner.create_error
        return FakeStream(self._owner.items, self._owner.delay)


class _FakeClient:
    def __init__(
        self,
        items: Iterable[Any] = (),
        *,
        delay: float = 0.0,
        create_error: Optional[BaseException] = None,
    ) -> None:
        self.items = list(items)
        self.delay = delay
        self.create_error = create_error
        self.calls: list[dict[str, Any]] = []

    async def close(self) -> None:
        return None


class FakeOpenAI(_FakeClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.chat = SimpleNamespace(completions=_Endpoint(self))


class FakeAnthropic(_FakeClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.messages = _Endpoint(self)


# --------------------------------------------------------------------------- #
# Event builders (shapes the gateway reads from each SDK)
# --------------------------------------------------------------------------- #
def oa_text(text: Optional[str]) -> SimpleNamespace:
    delta = SimpleNamespace(content=text)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def oa_role() -> SimpleNamespace:  # first chunk: role only, no content
    return oa_text(None)


def oa_finish() -> SimpleNamespace:
    delta = SimpleNamespace(content=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta, finish_reason="stop")])


def oa_usage(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(usage=usage, choices=[])  # final chunk: empty choices


def an_start(input_tokens: int, output_tokens: int = 1) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))


def an_text(text: str) -> SimpleNamespace:
    delta = SimpleNamespace(type="text_delta", text=text)
    return SimpleNamespace(type="content_block_delta", delta=delta)


def an_thinking() -> SimpleNamespace:  # a non-text delta the gateway must ignore
    delta = SimpleNamespace(type="thinking_delta", thinking="hmm")
    return SimpleNamespace(type="content_block_delta", delta=delta)


def an_delta(output_tokens: int) -> SimpleNamespace:  # cumulative count, no input_tokens attr
    return SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=output_tokens))


def an_event(kind: str) -> SimpleNamespace:
    return SimpleNamespace(type=kind)


def openai_script(tokens, input_tokens, output_tokens, *, with_usage: bool = True) -> list:
    items = [oa_role(), *(oa_text(t) for t in tokens), oa_finish()]
    if with_usage:
        items.append(oa_usage(input_tokens, output_tokens))
    return items


def anthropic_script(tokens, input_tokens, output_tokens) -> list:
    return [
        an_start(input_tokens),
        an_event("content_block_start"),
        an_thinking(),
        *(an_text(t) for t in tokens),
        an_event("content_block_stop"),
        an_delta(output_tokens),
        an_event("message_stop"),
    ]


def first_token(provider: str) -> SimpleNamespace:
    return oa_text("par") if provider == "openai" else an_text("par")


# --------------------------------------------------------------------------- #
# Gateway factories and helpers
# --------------------------------------------------------------------------- #
def make_gateway(
    *,
    openai_client: Any = None,
    anthropic_client: Any = None,
    openai_key: str = FAKE_OPENAI_KEY,
    anthropic_key: str = FAKE_ANTHROPIC_KEY,
) -> UnifiedLLMGateway:
    gw = UnifiedLLMGateway(openai_api_key=openai_key, anthropic_api_key=anthropic_key)
    if openai_client is not None:
        gw._openai = openai_client
    if anthropic_client is not None:
        gw._anthropic = anthropic_client
    return gw


def build(
    provider: str,
    *,
    items: Optional[list] = None,
    create_error: Optional[BaseException] = None,
    tokens=TOKENS,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    delay: float = 0.0,
    openai_key: str = FAKE_OPENAI_KEY,
    anthropic_key: str = FAKE_ANTHROPIC_KEY,
):
    """Return (gateway, model, fake_client) wired for one provider."""
    keys = {"openai_key": openai_key, "anthropic_key": anthropic_key}
    if provider == "openai":
        script = items if items is not None else openai_script(tokens, input_tokens, output_tokens)
        client = FakeOpenAI(script, delay=delay, create_error=create_error)
        return make_gateway(openai_client=client, **keys), OPENAI_MODEL, client
    script = items if items is not None else anthropic_script(tokens, input_tokens, output_tokens)
    client = FakeAnthropic(script, delay=delay, create_error=create_error)
    return make_gateway(anthropic_client=client, **keys), ANTHROPIC_MODEL, client


def run_stream(gw: UnifiedLLMGateway, prompt: str, model: str, temperature: float = 0.7) -> list[StreamChunk]:
    async def _collect() -> list[StreamChunk]:
        return [chunk async for chunk in gw.stream(prompt, model, temperature)]

    return asyncio.run(_collect())


# --------------------------------------------------------------------------- #
# Real-SDK exception builders (so error translation is tested against real classes)
# --------------------------------------------------------------------------- #
def _sdk(provider: str):
    return openai if provider == "openai" else anthropic


def _request(provider: str):
    # Built from the HTTP library the installed SDK is based on (httpx2 for openai>=3 / anthropic>=1).
    return http_module(provider).Request("POST", "https://api.invalid/v1/chat")


def sdk_status_error(provider, class_name, status, message, *, headers=None, body=None):
    response = http_module(provider).Response(status, headers=headers or {}, request=_request(provider))
    return getattr(_sdk(provider), class_name)(message, response=response, body=body)


def sdk_connection_error(provider, message="Connection error.", *, timeout=False):
    sdk = _sdk(provider)
    if timeout:
        return sdk.APITimeoutError(request=_request(provider))
    return sdk.APIConnectionError(message=message, request=_request(provider))