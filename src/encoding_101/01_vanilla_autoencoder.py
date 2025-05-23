import os
from pathlib import Path

import typer
from loguru import logger

from src.encoding_101.data import CIFAR10DataModule
from src.encoding_101.models import CIFAR10_CLASSES, VanillaAutoencoder
from src.encoding_101.training import train_autoencoder
from src.encoding_101.visualization import visualize_model_mar

ROOT_DIR = Path(__file__).parents[2]

app = typer.Typer()


@app.command()
def train_ae(
    data_dir: str = "./data",
    latent_dim: int = 128,
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    max_epochs: int = 100,
    debug: bool = True,
    visualize_mar: bool = True,
    mar_viz_epochs: int = 5,
    mar_samples_per_class: int = 5,
):
    """
    Train a vanilla autoencoder on CIFAR-10 dataset
    
    Args:
        data_dir: Directory where the data will be stored
        latent_dim: Dimensionality of the latent space
        batch_size: Batch size for training and validation
        num_workers: Number of workers for DataLoader
        device_id: GPU device ID to use
        max_epochs: Maximum number of epochs to train
        debug: Whether to run in debug mode
        visualize_mar: Whether to visualize MAR@5 during training
        mar_viz_epochs: How often to generate MAR@5 visualizations
        mar_samples_per_class: Number of samples per class for MAR@5 visualization
    """
    model_kwargs = {
        "latent_dim": latent_dim,
        "visualize_mar": visualize_mar,
        "mar_viz_epochs": mar_viz_epochs,
        "mar_samples_per_class": mar_samples_per_class,
    }
    
    model, trainer = train_autoencoder(
        model_class=VanillaAutoencoder,
        model_kwargs=model_kwargs,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device_id=device_id,
        max_epochs=max_epochs,
        debug=debug,
    )
    
    return model, trainer


@app.command()
def download_cifar10(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
):
    """
    Test the CIFAR-10 DataModule
    """
    logger.info("Testing CIFAR10DataModule")
    data_module = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Download the dataset if needed
    data_module.prepare_data()
    
    # Set up the dataset
    data_module.setup()
    
    # Test the dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    
    # Get a batch to see the data
    images, labels = next(iter(train_loader))
    logger.info(f"Batch shape: {images.shape}")
    logger.info(f"Labels shape: {labels.shape}")


@app.command()
def visualize_mar(
    checkpoint_path: str,
    data_dir: str = "./data",
    output_dir: str = "./mar_visualizations",
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    k: int = 5,
    samples_per_class: int = 10,
):
    """
    Visualize MAR@k for a trained vanilla autoencoder model using CIFAR-10 dataset
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        data_dir: Directory where the data is stored
        output_dir: Directory to save visualization images
        batch_size: Batch size for evaluation
        num_workers: Number of workers for DataLoader
        device_id: GPU device ID to use
        k: Number of nearest neighbors to consider
        samples_per_class: Number of samples to visualize per class
    """
    mar_at_k = visualize_model_mar(
        checkpoint_path=checkpoint_path,
        model_class=VanillaAutoencoder,
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        device_id=device_id,
        k=k,
        samples_per_class=samples_per_class,
        class_names=CIFAR10_CLASSES,
    )
    
    return mar_at_k


if __name__ == "__main__":
    app()
