import os
from pathlib import Path

import torch
from loguru import logger

from ..data import CIFAR10DataModule
from ..metrics import visualize_mar_at_k


def visualize_model_mar(
    checkpoint_path: str,
    model_class,
    data_dir: str = "./data",
    output_dir: str = "./mar_visualizations",
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    k: int = 5,
    samples_per_class: int = 10,
    class_names: list = None,
):
    """
    Visualize MAR@k for a trained model using CIFAR-10 dataset
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        model_class: The model class to load
        data_dir: Directory where the data is stored
        output_dir: Directory to save visualization images
        batch_size: Batch size for evaluation
        num_workers: Number of workers for DataLoader
        device_id: GPU device ID to use
        k: Number of nearest neighbors to consider
        samples_per_class: Number of samples to visualize per class
        class_names: List of class names for visualization
    
    Returns:
        mar_at_k: The computed MAR@k score
    """
    if class_names is None:
        # CIFAR-10 class names
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    # Setup data module
    data_module = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Load data
    data_module.prepare_data()
    data_module.setup()
    
    # Get validation dataloader
    validation_dataloader = data_module.val_dataloader()
    
    # Device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = model_class.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize MAR@k
    logger.info(f"Visualizing MAR@{k} with {samples_per_class} samples per class...")
    mar_at_k = visualize_mar_at_k(
        dataloader=validation_dataloader,
        model=model,
        output_dir=output_dir,
        samples_per_class=samples_per_class,
        k=k,
        device=device,
        class_names=class_names
    )
    
    logger.info(f"Overall MAR@{k}: {mar_at_k:.4f}")
    logger.info(f"Visualizations saved to: {output_dir}")
    
    return mar_at_k 