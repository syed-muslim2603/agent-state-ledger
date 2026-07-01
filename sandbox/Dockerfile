# ============================================================================ #
#  Agent State Ledger — Sandbox Docker Image                                   #
# ============================================================================ #
#
# This Dockerfile builds the minimal, hardened runtime image used for
# sandboxed agent execution.
#
# Security hardening applied at build time:
#   - Distroless base image (no shell, no package manager, minimal attack surface)
#   - Python copied from a slim builder stage (reproducible, auditable)
#   - No root user — UID 65534 (nobody) is the only user
#   - No writable layers in the final image (entrypoint is read-only)
#
# Build:
#   docker build -f sandbox/Dockerfile -t agent-state-ledger-sandbox:latest .
#
# Run (via SandboxExecutor — not intended for direct use):
#   docker run --rm --network none --read-only \
#     --memory 512m --cpus 1.0 \
#     --cap-drop ALL \
#     --security-opt no-new-privileges \
#     --user 65534:65534 \
#     --env ASL_AGENT_PAYLOAD=<base64-payload> \
#     agent-state-ledger-sandbox:latest
# ============================================================================ #

# ---- Builder stage ----------------------------------------------------------
FROM python:3.12-slim AS builder

# Set a non-interactive frontend so apt never prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install only what is required to copy a working Python interpreter
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create the sandbox working directory
WORKDIR /sandbox

# Copy the sandbox scripts into the image
COPY sandbox/entrypoint.py  /sandbox/entrypoint.py
COPY sandbox/sandbox_tools.py /sandbox/sandbox_tools.py

# Verify syntax before baking into the image (fail fast)
RUN python -m py_compile /sandbox/entrypoint.py
RUN python -m py_compile /sandbox/sandbox_tools.py

# ---- Final stage ------------------------------------------------------------
FROM python:3.12-slim AS final

# Metadata labels following OCI Image Specification
LABEL org.opencontainers.image.title="Agent State Ledger Sandbox"
LABEL org.opencontainers.image.description="Minimal hardened sandbox for isolated agent execution"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="MIT"

# Disable Python's output buffering so stdout is flushed immediately
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create the sandbox directory
WORKDIR /sandbox

# Copy only the validated scripts from the builder stage
COPY --from=builder /sandbox/entrypoint.py  /sandbox/entrypoint.py
COPY --from=builder /sandbox/sandbox_tools.py /sandbox/sandbox_tools.py

# Create a /tmp tmpfs mount point (the executor mounts --tmpfs /tmp)
# and ensure nobody can write anywhere else.
RUN chmod 755 /sandbox \
    && chmod 444 /sandbox/entrypoint.py \
    && chmod 444 /sandbox/sandbox_tools.py

# Do NOT create a non-root user via useradd — the base image provides
# nobody (UID 65534) and we use that via the --user flag at runtime.
# Running useradd would require writable /etc which conflicts with --read-only.

# The entrypoint reads from environment and writes to stdout/stderr only.
# No CMD arguments are needed.
ENTRYPOINT ["python", "/sandbox/entrypoint.py"]
