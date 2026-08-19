# Stage 1: Build the Vite frontend client
FROM node:20-alpine AS frontend-builder

WORKDIR /app/client

# Copy frontend dependency manifests
COPY client/package*.json ./

# Install frontend dependencies
RUN npm ci

# Copy frontend source code and build production assets into /app/client/dist
COPY client/ ./
RUN npm run build


# Stage 2: Python runtime environment
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runner

WORKDIR /app

# Install runtime system dependencies for audio, WebRTC, ONNX runtime, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    libasound2 \
    libopus0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=7860 \
    HOST=0.0.0.0

# Copy project dependency definitions first for optimal layer caching
COPY pyproject.toml uv.lock README.md LICENSE ./

# Pre-install dependencies without the local project root
RUN uv sync --frozen --no-dev --all-extras --no-extra gstreamer --no-extra local --no-install-project

# Copy Pipecat source and bot script
COPY src/ ./src/
COPY bot_web.py ./

# Install the project itself into the environment
RUN uv sync --frozen --no-dev --all-extras --no-extra gstreamer --no-extra local

# Copy prebuilt frontend static files from Stage 1
COPY --from=frontend-builder /app/client/dist ./client/dist

# Expose WebRTC server port
EXPOSE 7860

# Add container health check against the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch the Pipecat Web Agent
CMD ["uv", "run", "python", "bot_web.py"]
