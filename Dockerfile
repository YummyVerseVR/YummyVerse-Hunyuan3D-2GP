FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

COPY src/ ./src
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync 

# --frozen --no-dev --no-install-project 

COPY src ./src
COPY settings ./settings

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "src/entry.py"]
