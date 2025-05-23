# Encoding 101 - Autoencoder Framework

This is a refactored and modular framework for training and evaluating autoencoders on CIFAR-10 dataset. The code has been split into separate modules for better organization and easier extension.

## Project Structure

```
src/encoding_101/
├── __init__.py                    # Main package imports
├── 01_vanilla_autoencoder.py      # Script for vanilla autoencoder
├── 02_conv_autoencoder.py         # Script for convolutional autoencoder (example)
├── data/                          # Data loading modules
│   ├── __init__.py
│   └── cifar10.py                 # CIFAR-10 DataModule
├── models/                        # Model definitions
│   ├── __init__.py
│   ├── base.py                    # Base autoencoder class
│   ├── vanilla_autoencoder.py     # Vanilla fully-connected autoencoder
│   └── conv_autoencoder.py        # Convolutional autoencoder (example)
├── training/                      # Training utilities
│   ├── __init__.py
│   └── trainer.py                 # Training functions
├── visualization/                 # Visualization utilities
│   ├── __init__.py
│   └── mar_viz.py                 # MAR@k visualization functions
├── metrics.py                     # Metric calculation functions
└── README.md                      # This file
```

## How to Add a New Model

Adding a new autoencoder model is straightforward:

1. **Create the model class** in `models/your_model.py`:

```python
from .base import BaseAutoencoder

class YourAutoencoder(BaseAutoencoder):
    def __init__(self, latent_dim: int = 128, **kwargs):
        super().__init__(latent_dim, **kwargs)
        
        # Define your encoder and decoder architectures
        self.encoder_net = ...
        self.decoder_net = ...
    
    def encode(self, x):
        return self.encoder_net(x)
    
    def decode(self, z):
        return self.decoder_net(z)
```

2. **Update** `models/__init__.py` to export your model:

```python
from .your_model import YourAutoencoder
__all__ = [..., "YourAutoencoder"]
```

3. **Create a training script** (optional) like `03_your_autoencoder.py`:

```python
from .models import YourAutoencoder
from .training import train_autoencoder

# Use the training function with your model
model, trainer = train_autoencoder(
    model_class=YourAutoencoder,
    model_kwargs={"latent_dim": 128},
    # ... other training parameters
)
```

## Key Components

### BaseAutoencoder Class

- Handles common functionality like training loops, validation, and visualization
- Automatic MAR@k calculation and logging
- TensorBoard integration for training/validation image comparisons
- Abstract methods `encode()` and `decode()` that subclasses must implement

### CIFAR10DataModule

- PyTorch Lightning DataModule for CIFAR-10
- Automatic train/validation split with exactly 100 samples per class in validation
- Data augmentation for training set
- Proper normalization and preprocessing

### Training Function

- Generic training function that works with any autoencoder model
- Handles GPU/CPU selection, callbacks, and logging
- Graceful interruption handling

### Visualization Tools

- MAR@k visualization with nearest neighbor analysis
- Class-wise performance breakdown
- Interactive visualizations saved to files and TensorBoard

## MAR@k Evaluation

Mean Average Recall@k (MAR@k) measures how well the learned embeddings capture semantic similarity:

- For each query image, find k nearest neighbors in embedding space
- Calculate recall: how many neighbors belong to the same class
- Average across all queries to get MAR@k

This metric helps evaluate whether the autoencoder learns meaningful representations for retrieval tasks.

## TensorBoard Monitoring

Training automatically logs to TensorBoard:

```bash
uv run tensorboard --logdir logs/
```

You'll see:
- Training and validation loss curves
- MAR@5 metrics over time
- Image reconstruction comparisons
- MAR@k visualization summaries (if enabled)
