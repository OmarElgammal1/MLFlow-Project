# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

# Install runtime dependencies first so the layer is cached across code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and prediction artifacts.
COPY src ./src
COPY model.pkl transformer.pkl ./

EXPOSE 8000

# Run as non-root for safety.
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

CMD ["uv", "run", "--no-sync", "litestar", "--app", "src.app:app", "run", \
     "--host", "0.0.0.0", "--port", "8000"]
