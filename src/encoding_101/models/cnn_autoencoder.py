from collections import OrderedDict

import torch
import torch.nn as nn

from .base import BaseAutoencoder


class CNNAutoencoder(BaseAutoencoder):
    """Simple CNN autoencoder with fully connected layers"""
    
    def __init__(self, latent_dim: int = 128, visualize_mar: bool = False, 
                 mar_viz_epochs: int = 5, mar_samples_per_class: int = 5):
        super().__init__(latent_dim, visualize_mar, mar_viz_epochs, mar_samples_per_class)
        
        self.encoder_net = nn.Sequential(OrderedDict([
            ("encoder_conv1", nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)),
            ("encoder_relu1", nn.ReLU()),
            ("encoder_conv2", nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ("encoder_relu2", nn.ReLU()),
            ("encoder_conv3", nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            ("encoder_relu3", nn.ReLU()),
            ("encoder_flatten", nn.Flatten()),
            ("latent_space", nn.Linear(128 * 32 * 32, self.latent_dim)),
        ]))
        
        self.decoder_net = nn.Sequential(OrderedDict([
            ("decoder_linear1", nn.Linear(self.latent_dim, 128 * 32 * 32)),
            ("decoder_relu1", nn.ReLU()),
            ("decoder_reshape", nn.Unflatten(1, (128, 32, 32))),
            ("decoder_conv1", nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)),
            ("decoder_relu2", nn.ReLU()),
            ("decoder_conv2", nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)),
            ("decoder_relu3", nn.ReLU()),
            ("decoder_conv3", nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)),
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