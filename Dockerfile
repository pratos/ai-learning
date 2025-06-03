# Multi-stage build: Use uv for dependency management, NVIDIA for runtime
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set working directory
WORKDIR /app

# Set environment variables for uv
ENV UV_LINK_MODE=copy

# Install dependencies first (for better caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Runtime stage: Use NVIDIA PyTorch for CUDA support
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Set working directory
WORKDIR /app

# Install system dependencies for NVTX and profiling
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY . /app

# Install dependencies in runtime stage with cache mount
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/profiling_output

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

# Place environment executables on PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["bash"] 