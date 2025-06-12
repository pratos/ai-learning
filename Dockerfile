# Use NVIDIA PyTorch NGC container with Blackwell support
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set working directory
WORKDIR /app

# Install system dependencies for development and profiling
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt for additional packages not in NGC container
COPY requirements.txt /app/

# Install additional Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/profiling_output

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Blackwell Architecture Support (B200 GPU)
# Uncomment next line to test PTX compatibility if you encounter CUDA kernel errors
# ENV CUDA_FORCE_PTX_JIT=1

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["bash"] 