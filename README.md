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
├── scripts/
│   └── profile_training.py # NVTX profiling script
├── learning/              # Learning materials and guides
└── logs/                  # Training logs and checkpoints
└── data/                  # All the data stored here
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
# Profile your training with NVTX annotations
nsys profile --trace=nvtx,cuda python scripts/profile_training.py profile

# Analyze results with Nsight Systems
nsys-ui <generated_file>.qdrep
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
nsys profile --duration=30 --trace=nvtx,cuda python scripts/profile_training.py profile

# Profile with CPU context switching (advanced)
nsys profile --trace=nvtx,cuda,osrt --sample=cpu python scripts/profile_training.py profile

# Export specific metrics
nsys profile --trace=nvtx,cuda --output=training_profile python scripts/profile_training.py profile
```

---

## 📚 Learning Resources

- **`learning/learn-nvtx-annotations.md`** - Comprehensive NVTX learning guide with exercises
- **`scripts/profile_training.py`** - Ready-to-use profiling scripts
- **[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)** - Visual profiler for analyzing results
- **[PyTorch Profiler Guide](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)** - PyTorch profiling integration

## 🎯 Key Features

### Model Architecture
- **Latent Space Encoding**: Configurable dimensionality (default: 128)
- **CIFAR-10 Optimization**: Optimized for 32x32x3 image reconstruction
- **PyTorch Lightning**: Modern training framework with automatic logging

### Evaluation Metrics
- **MAR@5**: Mean Average Recall at k=5 for embedding quality assessment
- **Reconstruction Loss**: MSE between input and reconstructed images
- **Visual Comparison**: Side-by-side original vs reconstructed image grids

---

**Need Help?** Check the comprehensive learning guide or run:
```bash
uv run python scripts/profile_training.py --help
```
