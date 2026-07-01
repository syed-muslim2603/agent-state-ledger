# ============================================================================ #
#  Agent State Ledger — Router Service Dockerfile                              #
# ============================================================================ #
#
# Multi-stage build that produces a minimal production image for the
# FastAPI State Router service.
#
# Stages:
#   builder — Installs Python dependencies into a virtual environment
#   final   — Copies the venv and source into a slim runtime image
#
# Build:
#   docker build -f Dockerfile -t agent-state-ledger-router:latest .
# ============================================================================ #

# ---- Builder stage ----------------------------------------------------------
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies required for some native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first so pip install is cached when only source changes
COPY pyproject.toml ./
COPY src/ src/

# Create a virtual environment and install the package with all dependencies
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install ".[dev]"

# ---- Final stage ------------------------------------------------------------
FROM python:3.12-slim AS final

LABEL org.opencontainers.image.title="Agent State Ledger Router"
LABEL org.opencontainers.image.description="FastAPI/FastMCP State Router for multi-agent context management"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for the router service
RUN useradd --uid 1001 --gid 0 --no-create-home --shell /bin/false asl

# Data directory for the snapshot SQLite database
RUN mkdir -p /data && chown 1001:0 /data

WORKDIR /app

# Copy the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY --from=builder /build/src /app/src

USER 1001

# Expose HTTP port (overridable via ASL_ROUTER_PORT)
EXPOSE 8000

# Expose Prometheus metrics port
EXPOSE 9090

# Health check — pings the /v1/health endpoint every 30 seconds
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" \
    || exit 1

CMD ["python", "-m", "agent_state_ledger.router.main"]
