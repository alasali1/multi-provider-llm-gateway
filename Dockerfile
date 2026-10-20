# syntax=docker/dockerfile:1
FROM python:3.11-slim

# uv (fast installer) copied from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

# CA certificates: openai>=3 verifies HTTPS with the operating system's trust store.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only runtime dependencies first, so this layer is cached until pyproject/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Cache tiktoken's tokenizer inside the image so the first OpenAI request needs no download.
# (If the build machine is offline this is skipped and tiktoken downloads it on first use instead.)
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" || echo "tiktoken prefetch skipped"

# Copy the project code (.dockerignore keeps out tests, .env, .venv, .git).
# Layout: app.py, gateway.py, pricing.py, schemas.py and static/ sit in the project root.
COPY . ./

# Never run as root. API keys are NOT baked into the image: they arrive as environment variables.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]