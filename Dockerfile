# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.33 AS uv
FROM python:3.13.9-slim

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY config ./config
COPY scripts ./scripts
COPY src ./src

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

CMD ["/app/.venv/bin/uvicorn", "server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
