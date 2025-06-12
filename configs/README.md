# Hydra Configuration Management

This directory contains the Hydra configuration structure for autoencoder training experiments. Hydra provides hierarchical configuration management with powerful composition and override capabilities.

## 📁 Directory Structure

```
configs/
├── config.yaml                    # Main config file
├── model/                         # Model configurations
│   ├── nvtx_vanilla_autoencoder.yaml
│   └── cnn_autoencoder.yaml
├── training/                      # Training configurations
│   ├── default.yaml
│   └── profiling.yaml
├── data/                         # Data configurations
│   └── cifar10.yaml
├── profiling/                    # Profiling configurations
│   └── default.yaml
└── experiment/                   # Pre-configured experiments
    ├── cnn_comparison.yaml
    └── latent_space_sweep.yaml
```

## 🚀 Basic Usage

### Default Training
```bash
# Using the new Hydra script
python scripts/hydra_training.py

# Or via bin/run
bin/run hydra-train
```

### Override Parameters
```bash
# Change model and training parameters
python scripts/hydra_training.py model=cnn_autoencoder training.max_epochs=20

# Override multiple parameters
python scripts/hydra_training.py model.latent_dim=256 training.batch_size=128 device_id=1
```

### Use Different Configurations
```bash
# Use profiling configuration
python scripts/hydra_training.py training=profiling

# Use a specific experiment configuration
python scripts/hydra_training.py --config-name experiment/cnn_comparison
```

## 🔥 Advanced Features

### Multi-Run Experiments
```bash
# Train multiple models
python scripts/hydra_training.py model=nvtx_vanilla_autoencoder,cnn_autoencoder --multirun

# Parameter sweeps
python scripts/hydra_training.py model.latent_dim=64,128,256,512 training.max_epochs=5,10 --multirun

# Latent space sweep experiment
python scripts/hydra_training.py --config-name experiment/latent_space_sweep model.latent_dim=64,128,256,512 --multirun
```

### Configuration Interpolation
Hydra supports powerful variable interpolation:

```yaml
# In config files
experiment_name: ${model.name}_${training.max_epochs}ep_${training.batch_size}bs
output_dir: logs/tensorboard_logs
model:
  latent_dim: 128
training:
  batch_size: ${model.latent_dim}  # Reference other config values
```

### Environment Variable Override
```bash
# Override from environment
export MODEL_LATENT_DIM=512
python scripts/hydra_training.py model.latent_dim=${oc.env:MODEL_LATENT_DIM}
```

## 📋 Configuration Groups

### Models (`model/`)
- `nvtx_vanilla_autoencoder.yaml` - NVTX-annotated vanilla autoencoder
- `cnn_autoencoder.yaml` - Convolutional autoencoder

### Training (`training/`)
- `default.yaml` - Standard training configuration
- `profiling.yaml` - Optimized for profiling runs

### Data (`data/`)
- `cifar10.yaml` - CIFAR-10 dataset configuration

### Experiments (`experiment/`)
- `cnn_comparison.yaml` - CNN architecture comparison
- `latent_space_sweep.yaml` - Latent dimension parameter sweep

## 🎯 Integration with bin/run

The traditional `bin/run` commands still work, but you can also use the new Hydra-powered commands:

```bash
# Traditional approach
bin/run train CNNAutoencoder 20 128

# New Hydra approach (more flexible)
bin/run hydra-train config 'model=cnn_autoencoder training.max_epochs=20 training.batch_size=128'

# Multi-run experiments
bin/run hydra-train config 'model=nvtx_vanilla_autoencoder,cnn_autoencoder training.max_epochs=5,10' true
```

## 💡 Benefits Over Current Approach

### 🎛️ **Hierarchical Configuration**
- Logical grouping of related parameters
- Easy composition of different components
- Clear separation of concerns

### 🔧 **Flexible Overrides**
```bash
# Current typer approach - limited
python scripts/profile_training.py --model-name CNN --batch-size 128 --max-epochs 20

# Hydra approach - unlimited flexibility
python scripts/hydra_training.py model=cnn_autoencoder training.batch_size=128 training.max_epochs=20 model.latent_dim=256
```

### 🚀 **Multi-Run Experiments**
```bash
# Current approach - requires scripting/loops
for model in CNN Vanilla; do
  for epochs in 5 10 20; do
    python scripts/profile_training.py --model-name $model --max-epochs $epochs
  done
done

# Hydra approach - single command
python scripts/hydra_training.py model=cnn_autoencoder,nvtx_vanilla_autoencoder training.max_epochs=5,10,20 --multirun
```

### 📊 **Automatic Logging & Reproducibility**
- All configurations automatically saved
- Easy reproduction with exact same config
- Experiment tracking built-in

### 🧩 **Configuration Composition**
```yaml
# Mix and match components
defaults:
  - model: cnn_autoencoder      # Use CNN model
  - training: profiling         # Use profiling settings
  - data: cifar10              # Use CIFAR-10 data
```

## 🔄 Migration Path

1. **Phase 1**: Add Hydra alongside existing typer scripts ✅
2. **Phase 2**: Gradually migrate experiments to Hydra configs
3. **Phase 3**: Deprecate typer scripts in favor of Hydra
4. **Phase 4**: Full Hydra integration with advanced features (sweepers, plugins)

## 📚 Common Patterns

### Creating New Model Configs
```yaml
# configs/model/my_new_model.yaml
# @package _global_
model:
  _target_: src.encoding_101.models.my_model.MyNewModel
  name: MyNewModel
  latent_dim: 128
  custom_param: 42
```

### Creating Experiment Configs
```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - override /model: my_new_model
  - override /training: default

experiment_name: my_experiment_${now:%Y%m%d_%H%M%S}
model:
  latent_dim: 512  # Override for this experiment
training:
  max_epochs: 50
```

### Local Development
```bash
# Quick local testing
bin/run hydra-train-local config 'training.max_epochs=1 model.visualize_mar=false'
```

This Hydra setup provides a much more scalable and maintainable approach to configuration management compared to the current typer-based approach! 