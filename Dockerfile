FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY agenttrust/ agenttrust/
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/leeyamin/agent-trust" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Capability contract validation for A2A-protocol agents"

RUN groupadd -g 65532 agenttrust && \
    useradd -u 65532 -g 65532 -m agenttrust && \
    mkdir /work && chown agenttrust:agenttrust /work

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    AGENTTRUST_WORK_DIR="/work"

USER agenttrust
WORKDIR /work

ENTRYPOINT ["agenttrust"]
