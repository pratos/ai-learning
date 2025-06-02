import io
from typing import Optional

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from loguru import logger


def create_comparison_grid(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    stage: str,
    current_epoch: int,
    title_prefix: Optional[str] = None
) -> torch.Tensor:
    """
    Create a side-by-side comparison grid of original and reconstructed images.
    
    Args:
        original_images: Tensor of original images
        reconstructed_images: Tensor of reconstructed images  
        stage: Stage name (e.g., "Training", "Validation")
        current_epoch: Current epoch number
        title_prefix: Optional prefix for the title
        
    Returns:
        Tensor representation of the comparison grid image
    """
    if len(original_images) == 0:
        logger.warning(f"No images provided for {stage} comparison grid")
        return None
        
    n_images = len(original_images)
    
    # Create a side-by-side comparison grid
    comparison = torch.cat([original_images, reconstructed_images], dim=0)
    grid = torchvision.utils.make_grid(
        comparison, 
        nrow=n_images, 
        normalize=True, 
        padding=10,  # Increase padding between images
        pad_value=1.0,  # White padding
        scale_each=True  # Scale each image independently for better contrast
    )
    
    # Create a larger figure and add a title using matplotlib
    fig = plt.figure(figsize=(12, 6))  # Larger figure size
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    
    # Add labels for original and reconstructed
    plt.text(0.5, 0.05, "Original (top) vs Reconstruction (bottom)", 
             ha="center", transform=fig.transFigure, fontsize=14)
    
    # Create title
    title = f"{stage} Samples - Epoch {current_epoch}"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=3.0)
    
    # Convert figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # Convert to PIL image and then to tensor
    image = PIL.Image.open(buf)
    image_tensor = torchvision.transforms.ToTensor()(image)
    
    logger.debug(f"Created {stage.lower()} comparison grid for epoch {current_epoch}")
    
    return image_tensor 