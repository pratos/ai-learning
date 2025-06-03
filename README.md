# AI Learning

Learnings from implementing deep learning models from scratch.

## 📁 Project Structure

```
ai-learning/
├── src/encoding_101/
│   ├── models/              # Autoencoder implementations
│   │   ├── base.py         # Base autoencoder class
│   │   ├── vanilla_autoencoder.py
│   │   └── nvtx_autoencoder.py  # NVTX-annotated version
│   ├── training/           # Training utilities
│   ├── data.py              # Data loading and preprocessing
│   ├── metrics.py         # Evaluation metrics (MAR@5)
│   └── visualization/     # Plotting and analysis tools
├── bin/
│   └── run                 # Docker runtime script (uv-powered)
├── scripts/
│   └── profile_training.py # NVTX profiling script
├── learning/              # Learning materials and guides
├── logs/                  # Training logs and checkpoints
├── data/                  # All the data stored here
├── pyproject.toml         # Project dependencies (uv managed)
└── uv.lock                # Locked dependencies
```

## 🐳 Docker Training (uv-Powered)

The project uses [uv](https://docs.astral.sh/uv/) for ultra-fast Python package management in Docker, providing 10-100x faster dependency installation compared to pip.

### Prerequisites

The `bin/run` script will automatically install missing dependencies, but you can also install them manually:

1. **Docker & Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/) 
   - *Auto-installed*: `bin/run` detects and installs Docker + Docker Compose automatically
2. **NVIDIA Docker Runtime**: [Install nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - *Auto-installed*: `bin/run` detects and installs NVIDIA Docker runtime automatically  
3. **GPU Support**: No host GPU drivers needed!
   - *Container-only*: All CUDA drivers and GPU libraries are provided by the NVIDIA PyTorch container
   - *Server-safe*: No destructive changes to your host system

> 💡 **Smart Installation**: Just run `bin/run` and it will guide you through any missing dependencies!
> 🛡️ **Server-Safe**: All GPU drivers stay safely contained within Docker containers!

### 🚀 Quick Start

```bash
# First run - auto-installs Docker and NVIDIA runtime if needed
bin/run train

# Train vanilla autoencoder (10 epochs, batch size 64)
bin/run train

# Train with NVTX profiling (5 epochs, batch size 64)  
bin/run train-nvtx

# Run full NVTX profiling session
bin/run profile

# Start TensorBoard
bin/run tensorboard

# Interactive development shell
bin/run shell
```

### 🔧 uv-Powered Features

```bash
# Sync dependencies (blazing fast!)
bin/run sync

# Add new dependencies
bin/run uv add numpy

# Run any uv command
bin/run uv --help

# Custom training parameters
bin/run train 20 128        # 20 epochs, batch size 128
bin/run train-nvtx 5 32     # NVTX training, 5 epochs, batch size 32
```

### 🏗️ Docker Architecture

Our Docker setup uses [official uv images](https://docs.astral.sh/uv/guides/integration/docker/) with several optimizations:

- **Base Image**: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- **Intermediate Layers**: Dependencies installed separately for optimal caching
- **Cache Mounts**: Build cache preserved between builds for faster rebuilds
- **Virtual Environment**: Properly managed with uv sync workflow

### Docker Services

- **`ai-learning`**: Main training container with GPU support
- **`tensorboard`**: TensorBoard visualization (port 6007)
- **`profiling`**: Dedicated NVTX profiling container

### Manual Docker Commands

```bash
# Manual docker compose commands
docker compose up -d tensorboard              # Start TensorBoard service
docker compose run --rm ai-learning bash      # Interactive shell
docker compose down                           # Stop all services

# Build optimizations
docker compose build --progress=plain         # See build progress
docker compose build --no-cache               # Force rebuild without cache
```

---

## 🔍 NVTX Profiling for Performance Optimization

NVTX (NVIDIA Tools Extension) annotations provide semantic information about your training loop, enabling detailed performance analysis and bottleneck identification.

### What NVTX Gives You

Without NVTX annotations, GPU profilers show you raw kernels with no context:
- `volta_sgemm_128x64_tn`, `cudnn_conv_forward_kernel`
- Anonymous blocks of computation time
- No indication of which operation belongs to forward pass, backward pass, or data loading

With NVTX annotations, you get:
- **Semantic labels**: "Forward Pass", "Backward Pass", "MAR@5 Computation"
- **Hierarchical structure**: Nested annotations showing operation relationships
- **Performance attribution**: Clear identification of time-consuming operations
- **Color-coded timeline**: Visual distinction between different operation types

### 🎨 NVTX Color Scheme

| Operation | Color | Purpose |
|-----------|-------|---------|
| Training Steps | Green | Forward/backward training passes |
| Validation | Blue | Validation forward passes |
| Forward Pass | Yellow | Model forward computation |
| Loss Computation | Orange | Loss calculation |
| Metrics | Purple | MAR@5 and other metrics |
| Visualization | Pink | Image processing and logging |
| Data Transfer | Cyan | CPU↔GPU data movement |
| Memory Ops | Magenta | Memory allocation tracking |

### 📊 Using NVTX Profiling

```bash
# With Docker (recommended)
bin/run profile

# Manual profiling in container
docker-compose run --rm profiling \
    nsys profile --trace=nvtx,cuda --output=/app/profiling_output/profile \
    uv run scripts/profile_training.py profile

# Analyze results with Nsight Systems
nsys-ui ./profiling_output/profile.qdrep
```

### 🔍 What to Look for in Profiling Results

#### Timeline Analysis
1. **GPU Utilization**: Should be >80% during forward/backward passes
2. **Data Loading**: Should not block training (overlap with computation)
3. **Memory Transfers**: Minimize CPU↔GPU transfers
4. **Synchronization Points**: Identify unnecessary `torch.cuda.synchronize()` calls

#### Performance Bottlenecks
- **Long data loading times**: Increase `num_workers` or optimize data pipeline
- **Low GPU utilization**: Check for CPU bottlenecks or small batch sizes
- **Memory issues**: Look for large allocations or memory fragmentation
- **Excessive visualization**: MAR@5 computation taking too long

#### NVTX Annotation Hierarchy
```
Training Step
├── Data Unpack          [should be fast]
├── Forward Pass         [main computation]
│   ├── Encoder Forward  [first half]
│   └── Decoder Forward  [second half]
├── Loss Computation     [should be minimal]
└── Validation (every epoch)
    ├── MAR@5 Computation [periodic, can be expensive]
    └── Visualization     [periodic, I/O bound]
```

### ⚠️ Common Issues

#### NVTX Not Showing
- **Problem**: Annotations don't appear in Nsight Systems
- **Solution**: Ensure you're using `--trace=nvtx` flag and NVTX package is installed

#### Performance Overhead
- **Problem**: NVTX annotations slow down training
- **Solution**: Use `enable_nvtx=False` in production or reduce annotation granularity

#### Missing GPU Metrics
- **Problem**: No GPU utilization/memory data
- **Solution**: Add `--gpu-metrics-device=0` to nsys command

### 🛠️ Advanced NVTX Configuration

#### Custom NVTX Annotations

```python
from src.encoding_101.models.nvtx_autoencoder import NVTXProfiler, NVTXColors

profiler = NVTXProfiler(enabled=True)

# Custom operation profiling
with profiler.annotate("Custom Operation", NVTXColors.METRICS):
    your_custom_function()

# Memory profiling
with profiler.profile_memory("Large Allocation"):
    large_tensor = torch.randn(10000, 10000).cuda()
```

#### Integration with Existing Models

```python
import nvtx
from contextlib import nullcontext

class YourModel(nn.Module):
    def __init__(self, enable_profiling=False):
        super().__init__()
        self.enable_profiling = enable_profiling
    
    def nvtx_annotate(self, name, color="white"):
        if self.enable_profiling:
            return nvtx.annotate(name, color=color)
        else:
            return nullcontext()
    
    def forward(self, x):
        with self.nvtx_annotate("Custom Forward", "yellow"):
            return self.layers(x)
```

### 🧪 Advanced Profiling Commands

```bash
# Profile training for 30 seconds only
nsys profile --duration=30 --trace=nvtx,cuda uv run scripts/profile_training.py profile

# Profile with CPU context switching (advanced)
nsys profile --trace=nvtx,cuda,osrt --sample=cpu uv run scripts/profile_training.py profile

# Export specific metrics
nsys profile --trace=nvtx,cuda --output=training_profile uv run scripts/profile_training.py profile
```

---

## 📚 Learning Resources

- **[uv Documentation](https://docs.astral.sh/uv/)** - Ultra-fast Python package manager
- **[uv Docker Integration](https://docs.astral.sh/uv/guides/integration/docker/)** - Best practices for uv in Docker
- **`scripts/profile_training.py`** - Ready-to-use profiling scripts
- **[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)** - Visual profiler for analyzing results
- **[PyTorch Profiler Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)** - PyTorch profiling integration

---

**Need Help?** Check the comprehensive learning guide or run:
```bash
uv run scripts/profile_training.py --help
# Or with Docker:
bin/run help
```
