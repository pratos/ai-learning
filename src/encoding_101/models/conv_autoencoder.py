import torch
import torch.nn as nn

from .base import BaseAutoencoder


class ConvAutoencoder(BaseAutoencoder):
    """Convolutional autoencoder for CIFAR-10 images"""
    
    def __init__(self, latent_dim: int = 128, visualize_mar: bool = False, 
                 mar_viz_epochs: int = 5, mar_samples_per_class: int = 5):
        super().__init__(latent_dim, visualize_mar, mar_viz_epochs, mar_samples_per_class)
        
        # Encoder: 32x32x3 -> latent_dim
        self.encoder = nn.Sequential(
            # 32x32x3 -> 16x16x32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 16x16x32 -> 8x8x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 8x8x64 -> 4x4x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 4x4x128 -> 2x2x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            
            # Flatten and reduce to latent dimension
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, latent_dim)
        )
        
        # Decoder: latent_dim -> 32x32x3
        self.decoder = nn.Sequential(
            # Expand from latent dimension
            nn.Linear(latent_dim, 256 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (256, 2, 2)),
            
            # 2x2x256 -> 4x4x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 4x4x128 -> 8x8x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 8x8x64 -> 16x16x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 16x16x32 -> 32x32x3
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input images"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to images"""
        return self.decoder(z) 