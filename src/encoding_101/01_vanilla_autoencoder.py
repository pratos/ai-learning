import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import PIL
import torch
import torch.nn as nn
import torchvision
import typer
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

ROOT_DIR = Path(__file__).parents[2]

app = typer.Typer()

class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = os.cpu_count() or 4,
        train_val_split: float = 0.8,
        seed: int = 42,
    ):
        """
        PyTorch Lightning DataModule for CIFAR-10 dataset.
        
        Args:
            data_dir: Directory where the data will be stored
            batch_size: Batch size for training and validation
            num_workers: Number of workers for DataLoader
            train_val_split: Percentage of training data to use for training
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.seed = seed
        
        # Define transformations
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
    def prepare_data(self):
        """Download data if needed. This method is called only from a single process."""
        logger.info("Preparing CIFAR-10 dataset (downloading if needed)...")
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)
        
    def setup(self, stage: Optional[str] = None):
        """Setup train and val datasets. This is called from every process."""
        # Load the full training dataset
        cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
        
        # Create indices for each class
        class_indices = [[] for _ in range(10)]
        for idx, (_, label) in enumerate(cifar_full):
            class_indices[label].append(idx)
        
        # For validation, take exactly 100 samples from each class
        # using a fixed seed to ensure consistency across runs
        val_indices = []
        train_indices = []
        
        # Set a fixed seed for deterministic validation set
        rng = torch.Generator().manual_seed(self.seed)
        
        for class_idx in range(10):
            # Shuffle class indices
            perm = torch.randperm(len(class_indices[class_idx]), generator=rng)
            class_indices_shuffled = [class_indices[class_idx][i] for i in perm]
            
            # Take first 100 for validation
            val_indices.extend(class_indices_shuffled[:100])
            
            # Take the rest for training
            train_indices.extend(class_indices_shuffled[100:])
        
        # Create the train and validation datasets using the indices
        self.cifar_train = torch.utils.data.Subset(cifar_full, train_indices)
        
        # For validation set, we want clean transformations (no augmentation)
        cifar_val = CIFAR10(self.data_dir, train=True, transform=self.transform_val)
        self.cifar_val = torch.utils.data.Subset(cifar_val, val_indices)
        
        logger.info(f"Training set size: {len(self.cifar_train)}, Validation set size: {len(self.cifar_val)}")
        logger.info("Validation set has exactly 100 images from each of the 10 classes (1000 total)")
    
    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
class VanillaAutoencoder(LightningModule):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.save_hyperparameters()
        
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
        
        # Store images for visualization at the end of each epoch
        self.train_imgs = []
        self.train_recon_imgs = []
        self.val_imgs = []
        self.val_recon_imgs = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_net(x)
        decoded_embeddings = self.decoder_net(z)
        batch_size = x.shape[0]
        return decoded_embeddings.view(batch_size, 3, 32, 32)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        
        # Store images from first batch for visualization at the end of the epoch
        if batch_idx == 0:
            # Take up to 8 images for visualization
            n_images = min(8, x.size(0))
            self.train_imgs = x[:n_images].detach().clone()
            self.train_recon_imgs = x_hat[:n_images].detach().clone()
            
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)
        
        # Store the first batch for visualization at the end of the epoch
        if batch_idx == 0:
            # Take up to 8 images for visualization
            n_images = min(8, x.size(0))
            self.val_imgs = x[:n_images].detach().clone()
            self.val_recon_imgs = x_hat[:n_images].detach().clone()
            
        return loss
    
    def on_train_epoch_end(self):
        """Log training images at the end of each training epoch"""
        if len(self.train_imgs) > 0:
            # Create a side-by-side comparison grid
            n_images = len(self.train_imgs)
            
            # Add text labels to distinguish original from reconstruction
            # Create a higher quality grid with more padding and a custom size
            comparison = torch.cat([self.train_imgs, self.train_recon_imgs], dim=0)
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
            plt.title(f"Training Samples - Epoch {self.current_epoch}", fontsize=16)
            plt.tight_layout(pad=3.0)
            
            # Convert figure to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            plt.close(fig)
            buf.seek(0)
            
            # Convert to PIL image and then to tensor
            image = PIL.Image.open(buf)
            image = torchvision.transforms.ToTensor()(image)
            
            # Log to TensorBoard with consistent tag name for slider effect
            self.logger.experiment.add_image("train_comparison", image, self.current_epoch)
        
    def on_validation_epoch_end(self):
        """Log validation images at the end of each validation epoch"""
        if len(self.val_imgs) > 0:
            # Create a side-by-side comparison grid
            n_images = len(self.val_imgs)
            
            # Create a higher quality grid with more padding and a custom size
            comparison = torch.cat([self.val_imgs, self.val_recon_imgs], dim=0)
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
            plt.title(f"Validation Samples - Epoch {self.current_epoch}", fontsize=16)
            plt.tight_layout(pad=3.0)
            
            # Convert figure to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            plt.close(fig)
            buf.seek(0)
            
            # Convert to PIL image and then to tensor
            image = PIL.Image.open(buf)
            image = torchvision.transforms.ToTensor()(image)
            
            # Log to TensorBoard with consistent tag name for slider effect
            self.logger.experiment.add_image("val_comparison", image, self.current_epoch)
        
@app.command()
def train_ae(
    data_dir: str = "./data",
    latent_dim: int = 128,
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    debug: bool = True,
):
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ]
    
    logger_dir = ROOT_DIR / "logs"
    logger_dir.mkdir(parents=True, exist_ok=True)
    tf_logger = TensorBoardLogger(save_dir=logger_dir)

    logger.info(f"TensorBoard logger active. Saved in directory: {tf_logger.save_dir}")
    
    trainer = Trainer(
        logger=tf_logger,
        accelerator="auto",
        devices=[device_id],
        max_epochs=20,
        num_nodes=1,
        callbacks=callbacks,
    )
    
    data_module = CIFAR10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Download the dataset if needed
    data_module.prepare_data()
    
    # Setup train, val, and test datasets
    data_module.setup()
    
    # Now we can access the dataloaders
    training_dataloader = data_module.train_dataloader()
    validation_dataloader = data_module.val_dataloader()
    
    logger.info("Starting training...")
    model = VanillaAutoencoder(latent_dim=latent_dim)
    
    # Print model architecture and parameter count
    logger.info(f"Model Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Size: {trainable_params:,} trainable parameters ({total_params:,} total)")
    
    # Handle keyboard interrupts gracefully
    try:
        trainer.fit(
            model,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
        )
        logger.info("Training complete.")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Shutting down gracefully...")
        # Close dataloaders explicitly to avoid worker issues
        if hasattr(training_dataloader, '_iterator'):
            training_dataloader._iterator = None
        if hasattr(validation_dataloader, '_iterator'):
            validation_dataloader._iterator = None
        logger.info("Resources cleaned up. Exiting.")

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

if __name__ == "__main__":
    app()
