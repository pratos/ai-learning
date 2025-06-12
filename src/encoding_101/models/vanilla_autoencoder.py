from collections import OrderedDict

import torch
import torch.nn as nn

from .base import BaseAutoencoder
from src.encoding_101.mixins.nvtx import NVTXProfilingMixin

class VanillaAutoencoder(BaseAutoencoder):
    """Simple vanilla autoencoder with fully connected layers"""
    
    def __init__(self, latent_dim: int = 128, visualize_mar: bool = False, 
                 mar_viz_epochs: int = 5, mar_samples_per_class: int = 5):
        super().__init__(latent_dim, visualize_mar, mar_viz_epochs, mar_samples_per_class)
        
        self.encoder_net = nn.Sequential(OrderedDict([
            ("encoder_flatten", nn.Flatten()),
            ("encoder_linear1", nn.Linear(32 * 32 * 3, 1024)),
            ("encoder_relu1", nn.ReLU()),
            ("latent_space", nn.Linear(1024, self.latent_dim)),
        ]))
        
        self.decoder_net = nn.Sequential(OrderedDict([
            ("decoder_linear1", nn.Linear(self.latent_dim, 1024)),
            ("decoder_relu1", nn.ReLU()),
            ("decoder_linear2", nn.Linear(1024, 32 * 32 * 3)),
            ("decoder_sigmoid", nn.Sigmoid()),
        ]))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input images"""
        return self.encoder_net(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to images"""
        decoded = self.decoder_net(z)
        batch_size = z.shape[0]
        return decoded.view(batch_size, 3, 32, 32)

class NVTXVanillaAutoencoder(NVTXProfilingMixin, VanillaAutoencoder):
    """Vanilla Autoencoder with comprehensive NVTX profiling annotations"""

    def __init__(
        self,
        latent_dim: int = 128,
        visualize_mar: bool = False,
        mar_viz_epochs: int = 5,
        mar_samples_per_class: int = 5,
        enable_nvtx: bool = True,
    ):
        super().__init__(
            latent_dim=latent_dim,
            visualize_mar=visualize_mar,
            mar_viz_epochs=mar_viz_epochs,
            mar_samples_per_class=mar_samples_per_class,
            enable_nvtx=enable_nvtx,
        )