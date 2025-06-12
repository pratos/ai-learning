# AI Learning

Learnings from implementing deep learning models from scratch with configuration management (using Hydra) and NVTX profiling.

## 📁 Project Structure

```
ai-learning/
├── src/encoding_101/
│   ├── models/              # Autoencoder implementations
│   │   ├── base.py         # Base autoencoder class with MAR@5 visualization
│   │   ├── vanilla_autoencoder.py  # Vanilla + NVTX-enabled autoencoders
│   │   └── cnn_autoencoder.py      # CNN + NVTX-enabled autoencoders
│   ├── mixins/             # Reusable functionality
│   │   └── nvtx.py         # NVTX profiling mixin for any model
│   ├── training/           # Training utilities
│   │   └── trainer.py      # Main training orchestration
│   ├── data.py             # CIFAR-10 data module (Lightning)
│   ├── metrics.py          # Evaluation metrics (MAR@5)
│   ├── utils.py            # Visualization utilities
│   └── visualization/      # Advanced plotting and analysis tools
├── configs/                # Hydra configuration management
│   ├── config.yaml         # Main configuration with defaults
│   ├── model/              # Model configurations
│   │   ├── nvtx_vanilla_autoencoder.yaml
│   │   └── cnn_autoencoder.yaml
│   ├── training/           # Training configurations
│   │   ├── default.yaml
│   │   └── profiling.yaml
│   ├── data/               # Data module configurations
│   │   └── cifar10.yaml
│   ├── profiling/          # NVTX profiling settings
│   │   └── default.yaml
│   └── experiment/         # Pre-configured experiments
│       └── cnn_comparison.yaml
├── scripts/
│   ├── hydra_training.py   # Hydra-powered training script (pure loguru)
│   └── test_hydra_config.py # Configuration validation
├── bin/
│   └── run                 # Docker runtime script (uv-powered)
├── logs/                   # Comprehensive logging system
│   ├── hydra_runs/         # Default Hydra runs with full logs
│   ├── experiments/        # Experiment-specific runs
│   └── tensorboard_logs/   # TensorBoard visualization data
├── data/                   # CIFAR-10 dataset storage
├── pyproject.toml          # Project dependencies (uv managed)
└── uv.lock                 # Locked dependencies
```

## 🎯 Hydra Configuration Management

This project uses [Hydra](https://hydra.cc/) for configuration management, enabling:
- **Hierarchical configurations** with composable components
- **Experiment tracking** with automatic logging
- **Parameter sweeps** and multi-run experiments
- **Reproducible research** with complete configuration capture

### 🚀 Quick Start with Hydra

```bash
# Local training (no Docker) - recommended for development
bin/run train-local

# Override specific parameters
bin/run train-local model=cnn_autoencoder training.max_epochs=20

# Run pre-configured experiments
bin/run train-local --config-name=experiment/cnn_comparison

# Multi-run parameter sweeps
bin/run train-local model=nvtx_vanilla_autoencoder,cnn_autoencoder training.max_epochs=5,10,20 --multirun

# Profiling mode
bin/run train-local training=profiling

# Direct script usage (alternative)
uv run python scripts/hydra_training.py
uv run python scripts/hydra_training.py model=cnn_autoencoder training.max_epochs=20
```

### 📊 Configuration Examples

#### Model Configurations
```yaml
# configs/model/cnn_autoencoder.yaml
model:
  _target_: src.encoding_101.models.cnn_autoencoder.NVTXCNNAutoencoder
  latent_dim: 128
  visualize_mar: true
  enable_nvtx: true
```

#### Training Configurations
```yaml
# configs/training/default.yaml
max_epochs: 10
batch_size: 64
learning_rate: 1e-4
trainer:
  accelerator: auto
  devices: [0]
  precision: 16-mixed
```

#### Experiment Configurations
```yaml
# configs/experiment/cnn_comparison.yaml
defaults:
  - /model: cnn_autoencoder
  - /training: default

model:
  latent_dim: 256  # Larger latent space
training:
  max_epochs: 20
  batch_size: 128
```

## 📝 Pure Loguru Logging System

### 📁 Log Structure
```
logs/
├── hydra_runs/                    # Default configuration runs
│   └── [experiment_name]/
│       ├── .hydra/               # Complete Hydra metadata
│       │   ├── config.yaml       # Resolved configuration
│       │   ├── overrides.yaml    # CLI overrides used
│       │   └── hydra.yaml        # Hydra internal settings
│       └── [experiment_name].log # Full training logs
├── experiments/                   # Custom experiment runs
│   └── [experiment_name]/        # Same structure as above
└── tensorboard_logs/             # TensorBoard data
```

### 🎨 Log Output Examples
```
📝 Loguru logging configured - logs will be saved to: /path/to/log
🔮 Hydra Configuration:
🏗️ Instantiating model...
✅ Model class loaded: NVTXCNNAutoencoder
📊 Instantiating data module...
✅ Data module loaded: CIFAR10DataModule
🚀 Starting training process...
✅ Training complete!
```

## 🚀 Training Options

This project supports both **local training** (recommended for development) and **Docker training** (for full containerization).

### 🏠 Local Training (Recommended)
- **Faster iteration**: No Docker overhead
- **Direct access**: Work with your local environment
- **Easy debugging**: Direct Python debugging
- **Resource efficient**: Uses your system resources directly

```bash
# Setup once (installs uv and dependencies)
bin/run setup

# Train locally with Hydra
bin/run train-local
bin/run train-local model=cnn_autoencoder training.max_epochs=20
```

### 🐳 Docker Training (Full Containerization)

The project uses [uv](https://docs.astral.sh/uv/) for ultra-fast Python package management in Docker, providing 10-100x faster dependency installation compared to pip.

- **Consistent environment**: Same environment everywhere
- **GPU isolation**: Containerized CUDA drivers
- **Production-ready**: Matches deployment environment
- **Full profiling**: Complete NVTX profiling support

```bash
# Train with Docker
bin/run hydra-train
bin/run hydra-train model=cnn_autoencoder training.max_epochs=20
```

### Prerequisites

#### For Local Training (Recommended)
```bash
# One-time setup - installs uv and all dependencies
bin/run setup
```

#### For Docker Training (Optional)
The `bin/run` script will automatically install missing Docker dependencies:

1. **Docker & Docker Compose**: [Install Docker](https://docs.docker.com/get-docker/) 
   - *Auto-installed*: `bin/run` detects and installs Docker + Docker Compose automatically
2. **NVIDIA Docker Runtime**: [Install nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - *Auto-installed*: `bin/run` detects and installs NVIDIA Docker runtime automatically  
3. **GPU Support**: No host GPU drivers needed!
   - *Container-only*: All CUDA drivers and GPU libraries are provided by the NVIDIA PyTorch container
   - *Server-safe*: No destructive changes to your host system

> 💡 **Quick Start**: For most users, just run `bin/run setup` then `bin/run train-local`!
> 🐳 **Docker Users**: Run any `bin/run hydra-train` command and it will guide you through Docker setup!
> 🛡️ **Server-Safe**: All GPU drivers stay safely contained within Docker containers!

### 🏠 Local Development Quick Start

```bash
# One-time setup (installs uv and dependencies)
bin/run setup

# Basic training
bin/run train-local

# Run experiments
bin/run train-local --config-name=experiment/cnn_comparison

# Override parameters
bin/run train-local model=cnn_autoencoder training.max_epochs=20

# Multi-run parameter sweeps
bin/run train-local model=nvtx_vanilla_autoencoder,cnn_autoencoder --multirun

# Start TensorBoard
bin/run tensorboard-uv

# List available models
bin/run list-models-local
```

### 🐳 Docker Quick Start

```bash
# Basic training with Docker
bin/run hydra-train

# Run experiments with Docker
bin/run hydra-train --config-name=experiment/cnn_comparison

# Override parameters with Docker
bin/run hydra-train model=cnn_autoencoder training.max_epochs=20

# Multi-run parameter sweeps with Docker
bin/run hydra-train model=nvtx_vanilla_autoencoder,cnn_autoencoder --multirun

# Start TensorBoard with Docker
bin/run tensorboard

# Interactive development shell
bin/run shell

# List available models
bin/run list-models
```

### 🔧 Advanced Features

```bash
# Setup local development environment (required for local training)
bin/run setup

# Legacy training commands (still supported for backwards compatibility)
bin/run train 20 128        # Legacy: 20 epochs, batch size 128 (Docker)
bin/run train-nvtx 5 32     # Legacy: NVTX training, 5 epochs, batch size 32 (Docker)

# Development tools
bin/run shell               # Interactive development shell (Docker)
bin/run cleanup             # Clean up Docker resources
bin/run export-requirements # Export filtered requirements.txt
bin/run list-models         # List available models (Docker)
bin/run list-models-local   # List available models (local)
```

## 🔍 NVTX Profiling for Performance Optimization

NVTX (NVIDIA Tools Extension) annotations provide semantic information about your training loop, enabling detailed performance analysis and bottleneck identification.

### 🎨 NVTX Integration

All models support NVTX profiling through the `NVTXProfilingMixin`:

```python
# Available NVTX-enabled models
from src.encoding_101.models.vanilla_autoencoder import NVTXVanillaAutoencoder
from src.encoding_101.models.cnn_autoencoder import NVTXCNNAutoencoder

# NVTX can be enabled/disabled per model
model = NVTXVanillaAutoencoder(enable_nvtx=True)
```

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
# With Hydra configuration (local)
bin/run train-local training=profiling

# With Hydra configuration (Docker)
bin/run hydra-train training=profiling

# Direct script usage (local)
uv run python scripts/hydra_training.py training=profiling

# Legacy profiling with Docker (still supported)
bin/run profile

# Manual profiling in container
docker-compose run --rm profiling \
    nsys profile --trace=nvtx,cuda --output=/app/profiling_output/profile \
    uv run scripts/hydra_training.py training=profiling

# Analyze results with local Mac/Windows system. Download the file and load up the *.qdrep file
```

#### NVTX Annotation Hierarchy
```
🚀 EPOCH N - TRAINING START
├── Training Step
│   ├── Data Unpack          [should be fast]
│   ├── Forward Pass         [main computation]
│   │   ├── Encoder Forward  [first half]
│   │   └── Decoder Forward  [second half]
│   └── Loss Computation     [should be minimal]
🔍 EPOCH N - VALIDATION START
├── Validation Step
│   ├── Val Forward Pass
│   ├── MAR@5 Computation    [periodic, can be expensive]
│   └── Visualization        [periodic, I/O bound]
✅ EPOCH N - VALIDATION END
```

## 🧪 Advanced Usage

### Multi-Run Experiments

```bash
# Compare different models (local - recommended)
bin/run train-local model=nvtx_vanilla_autoencoder,cnn_autoencoder --multirun

# Compare different models (Docker)
bin/run hydra-train model=nvtx_vanilla_autoencoder,cnn_autoencoder --multirun

# Parameter sweeps (local - recommended)
bin/run train-local training.learning_rate=1e-3,1e-4,1e-5 model.latent_dim=64,128,256 --multirun

# Parameter sweeps (Docker)
bin/run hydra-train training.learning_rate=1e-3,1e-4,1e-5 model.latent_dim=64,128,256 --multirun

# Batch size optimization (local)
bin/run train-local training.batch_size=32,64,128,256 --multirun

# Batch size optimization (Docker)
bin/run hydra-train training.batch_size=32,64,128,256 --multirun

# Direct script usage (alternative)
uv run python scripts/hydra_training.py model=nvtx_vanilla_autoencoder,cnn_autoencoder --multirun
```

### Custom Experiments

Create your own experiment configurations in `configs/experiment/`:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - /model: cnn_autoencoder
  - /training: default
  - _self_

experiment_name: my_custom_experiment_${now:%Y%m%d_%H%M%S}

model:
  latent_dim: 512
  visualize_mar: true
  mar_viz_epochs: 1

training:
  max_epochs: 50
  batch_size: 256
  learning_rate: 5e-4

hydra:
  run:
    dir: logs/experiments/${experiment_name}
```

### Configuration Validation

```bash
# Test configuration without training (local)
bin/run train-local --cfg job

# Test configuration without training (Docker)
bin/run hydra-train --cfg job

# Validate specific experiment (local)
bin/run train-local --config-name=experiment/my_experiment --cfg job

# Validate specific experiment (Docker)
bin/run hydra-train --config-name=experiment/my_experiment --cfg job

# Direct script usage (alternative)
uv run python scripts/test_hydra_config.py
uv run python scripts/hydra_training.py --config-name=experiment/my_experiment --cfg job
```

## 📚 Learning Resources

- **[Hydra Documentation](https://hydra.cc/)** - Configuration management framework
- **[Loguru Documentation](https://loguru.readthedocs.io/)** - Beautiful Python logging
- **[uv Documentation](https://docs.astral.sh/uv/)** - Ultra-fast Python package manager
- **[uv Docker Integration](https://docs.astral.sh/uv/guides/integration/docker/)** - Best practices for uv in Docker
- **[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)** - Visual profiler for analyzing results
- **[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)** - Training framework used

## 🎯 Key Features

### ✅ Configuration Management
- **Hierarchical configs** with Hydra
- **Experiment tracking** with automatic logging
- **Parameter sweeps** and multi-run support
- **Complete reproducibility** with config capture

### ✅ Logging & Monitoring
- **Pure loguru logging** with beautiful output
- **Comprehensive log capture** (training, metrics, errors)
- **Automatic log rotation** and compression
- **TensorBoard integration** for visualization

### ✅ Performance Profiling
- **NVTX annotations** for semantic profiling
- **Mixin-based design** for easy integration
- **Color-coded timeline** analysis
- **GPU utilization** monitoring

### ✅ Model Architecture
- **Modular design** with mixins
- **NVTX-enabled models** for profiling
- **MAR@5 visualization** for embedding analysis
- **PyTorch Lightning** integration

---

**Need Help?** Check the configuration examples or run:
```bash
# Main help and commands
bin/run help

# Local training help
bin/run train-local --help

# Docker training help
bin/run hydra-train --help

# List available models (local)
bin/run list-models-local

# List available models (Docker)
bin/run list-models

# Direct script help (alternative)
uv run python scripts/hydra_training.py --help
```
