import os

import typer

from .models import CIFAR10_CLASSES, ConvAutoencoder
from .training import train_autoencoder
from .visualization import visualize_model_mar

app = typer.Typer()


@app.command()
def train_conv_ae(
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
    Train a convolutional autoencoder on CIFAR-10 dataset
    """
    model_kwargs = {
        "latent_dim": latent_dim,
        "visualize_mar": visualize_mar,
        "mar_viz_epochs": mar_viz_epochs,
        "mar_samples_per_class": mar_samples_per_class,
    }
    
    model, trainer = train_autoencoder(
        model_class=ConvAutoencoder,
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
def visualize_conv_mar(
    checkpoint_path: str,
    data_dir: str = "./data",
    output_dir: str = "./conv_mar_visualizations",
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    k: int = 5,
    samples_per_class: int = 10,
):
    """
    Visualize MAR@k for a trained convolutional autoencoder model
    """
    mar_at_k = visualize_model_mar(
        checkpoint_path=checkpoint_path,
        model_class=ConvAutoencoder,
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