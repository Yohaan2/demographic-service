# ==============================================================================
# STAGE 1: Compile Frontend Assets (React + Vite)
# ==============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/client

# Copy package descriptors and download Node packages
COPY client/package*.json ./
RUN npm install

# Copy source and build static distribution directory (dist/)
COPY client/ ./
RUN npm run build


# ==============================================================================
# STAGE 2: Python Backend Dependency Builder
# ==============================================================================
FROM python:3.12-slim AS backend-builder

WORKDIR /app

# Ensure compilation tools are available (build-essential for compiled wheels, git for VCS installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies system-wide (readable/executable by any container user)
COPY requirements.insightface.txt .
# Install only the InsightFace-compatible lightweight stack (no PyTorch, no MiVOLO)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.insightface.txt


# ==============================================================================
# STAGE 3: Single Production Image Runner
# ==============================================================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and executables from builder stage (system-wide location)
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# NOTE: MiVOLO / PyTorch support is disabled in this image.
# To re-enable, switch back to requirements.txt and restore the mivolo install step.
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy compiled static frontend assets from Node build stage
COPY --from=frontend-builder /app/client/dist ./client/dist

# Create models weights directory structure
RUN mkdir -p models/weights

# Copy backend codebase
COPY src/ ./src/

# Set up non-root security privileges and a writable HuggingFace cache directory
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appgroup /app
USER appuser

# Direct HuggingFace model downloads to a writable directory owned by appuser
ENV HF_HOME=/app/.cache/huggingface

# Expose API & Frontend consolidated PORT
EXPOSE 8000

# Python path configuration
ENV PYTHONPATH=/app

# Native, highly secure Python healthcheck (Zero vulnerability setup)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Launch FastAPI utilizing Gunicorn as process manager and Uvicorn for asynchronous requests
# WEB_CONCURRENCY defaults to 4 if not set (typical for 4-core containers)
CMD ["sh", "-c", "gunicorn src.main:app -w ${WEB_CONCURRENCY:-4} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --timeout 120"]
