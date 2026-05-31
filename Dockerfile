# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency builder
# Compile wheels in an isolated stage so the final image has no build toolchain.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build dependencies (only needed to compile certain C-extension wheels)

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
  && rm -rf /var/lib/apt/lists/*

  # Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    apt-transport-https \
    ca-certificates \
    gnupg

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash
WORKDIR /build

# Copy only requirements first to exploit Docker layer cache:
# this layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .

# Install into a local prefix so we can copy cleanly in the next stage.
# --extra-index-url points to the CPU-only PyTorch wheel registry.
RUN pip install --upgrade pip --no-cache-dir \
 && pip install \
    --no-cache-dir \
    --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# Lean python:3.11-slim with only the installed packages copied from the builder.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libraries required by OpenCV and the Azure SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# ── Non-root user ─────────────────────────────────────────────────────────────
# Running as non-root is a security best practice for containerised services.
ARG APP_USER=appuser
ARG APP_UID=1001
RUN useradd --uid ${APP_UID} --no-create-home --shell /bin/false ${APP_USER}

WORKDIR /app

# Copy application source
COPY --chown=${APP_USER}:${APP_USER} app/ ./app/

# ── Model file ────────────────────────────────────────────────────────────────
# The exported TorchScript model must be placed at:
#   app/ml/exported_model/siamese_signature_model.pt
# Either COPY it here (recommended for immutable images):
#   COPY --chown=${APP_USER}:${APP_USER} app/ml/exported_model/ ./app/ml/exported_model/
# Or mount it as an Azure File Share volume (for easy model updates without rebuild).
# The path is configurable via the MODEL_PATH environment variable.

USER ${APP_USER}

# ── Environment defaults ──────────────────────────────────────────────────────
# Non-secret defaults. Secrets (connection strings, keys) MUST be provided
# via Azure Container Apps secret environment variables — never baked into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    ENVIRONMENT=production

EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# Docker / Azure will restart the container if health checks fail.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
# --workers 1: ML models are not fork-safe; use a single worker.
# --loop uvloop: faster async event loop (included in uvicorn[standard]).
# --log-level warning: structlog handles application-level logging.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "warning", \
     "--no-access-log"]
