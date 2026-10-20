"""FastAPI + SSE web client for UnifiedLLMGateway.

Run:  uv run uvicorn app:app --reload
Open: http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gateway import UnifiedLLMGateway
from schemas import (
    AuthenticationFailedError,
    GatewayError,
    RateLimitExceededError,
    UnsupportedModelError,
)

logger = logging.getLogger("app")
BASE_DIR = Path(__file__).parent

# Models offered in the dropdown. Each ID must resolve in pricing.PRICING (prefix match).
# Anthropic needs IDs the API accepts; edit this list if a model is retired.
MODEL_CATALOG: dict[str, dict] = {
    "openai": {
        "max_temperature": 2.0,
        "models": ["gpt-4o-mini", "gpt-4o"],
    },
    "anthropic": {
        # Claude 5.x ignores this setting (the gateway does not send it; the API would answer 400).
        "max_temperature": 1.0,
        "models": ["claude-haiku-5-5", "claude-sonnet-5-5"],
    },
}


class StreamRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    provider: Literal["openai", "anthropic"]
    model: str
    prompt: str = Field(min_length=1, max_length=20_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with UnifiedLLMGateway() as gateway:  # keys come from env vars
        app.state.gateway = gateway
        yield


app = FastAPI(title="LLM Gateway Playground", lifespan=lifespan)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _event_stream(gateway: UnifiedLLMGateway, req: StreamRequest) -> AsyncIterator[str]:
    try:
        async for chunk in gateway.stream(req.prompt, req.model, req.temperature):
            if chunk.is_final:
                yield _sse("final", chunk.telemetry.model_dump())
            else:
                yield _sse("token", {"delta": chunk.delta_text})
    except RateLimitExceededError as exc:
        yield _sse(
            "error",
            {"type": "rate_limit", "message": str(exc), "retry_after_s": exc.retry_after_s},
        )
    except AuthenticationFailedError as exc:
        yield _sse("error", {"type": "auth", "message": str(exc)})
    except (GatewayError, ValueError) as exc:
        yield _sse("error", {"type": type(exc).__name__, "message": str(exc)})
    except Exception as exc:  # noqa: BLE001 - never leak internals to the browser
        logger.error("Unhandled stream failure: %s", type(exc).__name__)
        yield _sse("error", {"type": "internal", "message": "Unexpected server error."})


@app.get("/api/models")
async def list_models() -> dict:
    return MODEL_CATALOG


@app.post("/api/stream")
async def stream(req: StreamRequest) -> StreamingResponse:
    if req.model not in MODEL_CATALOG[req.provider]["models"]:
        raise HTTPException(status_code=400, detail="Model not available for this provider.")
    if req.temperature > MODEL_CATALOG[req.provider]["max_temperature"]:
        raise HTTPException(status_code=400, detail="Temperature too high for this provider.")
    return StreamingResponse(
        _event_stream(app.state.gateway, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")