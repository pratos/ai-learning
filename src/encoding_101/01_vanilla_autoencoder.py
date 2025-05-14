import io
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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

from src.encoding_101.metrics import visualize_mar_at_k

ROOT_DIR = Path(__file__).parents[2]

app = typer.Typer()

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

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
    def __init__(self, latent_dim: int = 128, visualize_mar: bool = False, 
                 mar_viz_epochs: int = 5, mar_samples_per_class: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.save_hyperparameters()
        
        # MAR visualization settings
        self.visualize_mar = visualize_mar
        self.mar_viz_epochs = mar_viz_epochs
        self.mar_samples_per_class = mar_samples_per_class
        
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
        
        # For MAR@5 calculation
        self.val_embeddings = []
        self.val_labels = []
        self.val_images = []
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input images"""
        return self.encoder_net(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to images"""
        decoded = self.decoder_net(z)
        batch_size = z.shape[0]
        return decoded.view(batch_size, 3, 32, 32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
    
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
        
        # Store embeddings, labels and images for MAR@5 calculation and visualization
        embeddings = self.encode(x)
        self.val_embeddings.append(embeddings.detach())
        self.val_labels.append(y.detach())
        
        # Only store images if we're going to visualize MAR
        if self.visualize_mar and self.current_epoch % self.mar_viz_epochs == 0:
            self.val_images.append(x.detach().cpu())
            
        return loss
    
    def on_validation_epoch_start(self):
        """Clear stored embeddings and labels at the start of validation"""
        self.val_embeddings = []
        self.val_labels = []
        if self.visualize_mar and self.current_epoch % self.mar_viz_epochs == 0:
            self.val_images = []
    
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
    
    def compute_mar_at_k(self, embeddings, labels, k=5):
        """
        Compute Mean Average Recall@k
        
        Args:
            embeddings: Tensor of shape (n, d) with n samples and d dimensions
            labels: Tensor of shape (n,) with class labels
            k: Number of nearest neighbors to consider
            
        Returns:
            mar_at_k: Mean Average Recall@k
        """
        # Compute pairwise distances between all embeddings
        # We use negative cosine similarity as our distance metric
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        cos_sim = torch.mm(normalized_embeddings, normalized_embeddings.t())
        
        # Set diagonal to -inf to exclude self-comparisons
        cos_sim.fill_diagonal_(-float('inf'))
        
        # Get top-k indices for each embedding
        _, topk_indices = cos_sim.topk(k=k, dim=1)
        
        # Compute recall@k for each query
        recalls = []
        
        # Convert labels to numpy for easier handling
        labels_np = labels.cpu().numpy()
        topk_indices_np = topk_indices.cpu().numpy()
        
        for i, query_label in enumerate(labels_np):
            # Get labels of the top-k nearest neighbors
            neighbor_labels = labels_np[topk_indices_np[i]]
            
            # Count how many are from the same class
            relevant_retrieved = (neighbor_labels == query_label).sum()
            
            # Count total number of relevant items in the dataset (excluding self)
            total_relevant = (labels_np == query_label).sum() - 1
            
            # Calculate recall for this query
            if total_relevant > 0:
                recall = min(relevant_retrieved / total_relevant, 1.0)
                recalls.append(recall)
        
        # Calculate mean recall
        if recalls:
            return sum(recalls) / len(recalls)
        else:
            return 0.0
    
    def on_validation_epoch_end(self):
        """Log validation images and compute MAR@5 at the end of each validation epoch"""
        # Process validation images
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
        
        # Compute MAR@5 if we have embeddings
        if self.val_embeddings:
            # Concatenate all stored embeddings and labels
            all_embeddings = torch.cat(self.val_embeddings, dim=0)
            all_labels = torch.cat(self.val_labels, dim=0)
            
            # Compute MAR@5
            mar_at_5 = self.compute_mar_at_k(all_embeddings, all_labels, k=5)
            
            # Log the metric
            self.log("val_mar_at_5", mar_at_5)
            
            # Also print it for visibility during training
            logger.info(f"Epoch {self.current_epoch}: MAR@5 = {mar_at_5:.4f}")
            
            # Visualize MAR@5 periodically during training if enabled
            if self.visualize_mar and self.current_epoch % self.mar_viz_epochs == 0 and len(self.val_images) > 0:
                # Concatenate all stored images
                all_images = torch.cat(self.val_images, dim=0)
                
                # Get the log directory for this run
                tensorboard_dir = self.logger.log_dir if hasattr(self.logger, 'log_dir') else None
                if tensorboard_dir:
                    mar_viz_dir = Path(tensorboard_dir) / f"mar_viz_epoch_{self.current_epoch}"
                    mar_viz_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Use the visualization function
                    logger.info(f"Visualizing MAR@5 for epoch {self.current_epoch}...")
                    
                    # Prepare the data for visualization
                    class_indices = {}
                    for i in range(10):  # Assuming 10 classes in CIFAR-10
                        class_indices[i] = (all_labels == i).nonzero(as_tuple=True)[0]
                    
                    # Select samples for each class
                    selected_indices = []
                    for class_idx in range(10):
                        indices = class_indices[class_idx]
                        if len(indices) > self.mar_samples_per_class:
                            # Randomly select samples
                            selected = indices[torch.randperm(len(indices))[:self.mar_samples_per_class]]
                            selected_indices.append(selected)
                        else:
                            # Use all available samples
                            selected_indices.append(indices)
                    
                    # Flatten the list
                    if selected_indices:
                        selected_indices = torch.cat(selected_indices)
                        
                        # Generate visualizations
                        self.visualize_mar_for_selected(
                            all_images, all_embeddings, all_labels, 
                            selected_indices, mar_viz_dir, k=5
                        )
                        
                        # Log a summary figure to TensorBoard
                        summary_path = mar_viz_dir / "mar_at_5_summary.png"
                        if summary_path.exists():
                            summary_img = PIL.Image.open(summary_path)
                            summary_tensor = torchvision.transforms.ToTensor()(summary_img)
                            self.logger.experiment.add_image(
                                "MAR@5_summary", summary_tensor, self.current_epoch
                            )
            
            # Clear for next epoch
            self.val_embeddings = []
            self.val_labels = []
            self.val_images = []
            
    def visualize_mar_for_selected(self, images, embeddings, labels, indices, output_dir, k=5):
        """Create visualizations for selected indices and save to output_dir"""
        # Normalize embeddings for similarity calculation
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        cos_sim = torch.mm(normalized_embeddings, normalized_embeddings.t())
        cos_sim.fill_diagonal_(-float('inf'))
        
        # Calculate recall per class
        class_recalls = {}
        total_recalls = []
        
        # Ensure all tensors are on the same device (CPU for visualization)
        images = images.cpu()
        embeddings = embeddings.cpu()
        labels = labels.cpu()
        indices = indices.cpu()
        
        # Process each query
        for i, query_idx in enumerate(indices):
            query_img = images[query_idx]
            query_label = labels[query_idx].item()
            
            # Get top-k nearest neighbors
            similarities = cos_sim[query_idx].cpu()  # Ensure on CPU
            _, nn_indices = similarities.topk(k)
            nn_labels = labels[nn_indices]
            nn_images = images[nn_indices]
            
            # Calculate recall
            total_relevant = (labels == query_label).sum().item() - 1  # Excluding self
            relevant_retrieved = (nn_labels == query_label).sum().item()
            recall = min(relevant_retrieved / total_relevant, 1.0) if total_relevant > 0 else 0
            total_recalls.append(recall)
            
            # Track recall by class
            if query_label not in class_recalls:
                class_recalls[query_label] = []
            class_recalls[query_label].append(recall)
            
            # Create visualization
            fig, axes = plt.subplots(1, k+1, figsize=(15, 3))
            
            # Query image
            query_img_np = query_img.permute(1, 2, 0).numpy()
            query_img_np = np.clip(query_img_np, 0, 1)
            axes[0].imshow(query_img_np)
            axes[0].set_title(f"Query\n{CIFAR10_CLASSES[query_label]}")
            axes[0].axis('off')
            
            # Neighbor images
            for j in range(k):
                nn_img = nn_images[j]
                nn_label = nn_labels[j].item()
                
                nn_img_np = nn_img.permute(1, 2, 0).numpy()
                nn_img_np = np.clip(nn_img_np, 0, 1)
                
                # Add red/green border based on whether it's the same class
                is_same_class = nn_label == query_label
                border_color = 'green' if is_same_class else 'red'
                
                # Display with colored border
                axes[j+1].imshow(nn_img_np)
                for spine in axes[j+1].spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(5)
                
                axes[j+1].set_title(f"NN {j+1}\n{CIFAR10_CLASSES[nn_label]}")
                axes[j+1].axis('off')
            
            plt.suptitle(f"Query: {CIFAR10_CLASSES[query_label]}, Recall@{k}: {recall:.2f}", fontsize=16)
            plt.tight_layout()
            
            # Save
            class_dir = Path(output_dir) / f"class_{query_label}"
            class_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(class_dir / f"sample_{i % self.mar_samples_per_class}.png", dpi=150)
            plt.close(fig)
        
        # Calculate overall MAR@k
        mar_at_k = sum(total_recalls) / len(total_recalls) if total_recalls else 0
        
        # Create summary bar chart
        plt.figure(figsize=(12, 6))
        classes = sorted(class_recalls.keys())
        class_mean_recalls = [np.mean(class_recalls[c]) for c in classes]
        
        bars = plt.bar(classes, class_mean_recalls)
        plt.title(f'MAR@{k} by Class (Overall: {mar_at_k:.4f})', fontsize=16)
        plt.xlabel('Class')
        plt.ylabel(f'Mean Recall@{k}')
        plt.xticks(classes, [CIFAR10_CLASSES[i] for i in classes], rotation=45)
        
        # Color bars by recall value
        for i, bar in enumerate(bars):
            if class_mean_recalls[i] > 0.8:
                bar.set_color('darkgreen')
            elif class_mean_recalls[i] > 0.6:
                bar.set_color('green')
            elif class_mean_recalls[i] > 0.4:
                bar.set_color('orange')
            elif class_mean_recalls[i] > 0.2:
                bar.set_color('darkorange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'mar_at_{k}_summary.png', dpi=150)
        plt.close()
        
        logger.info(f"MAR@{k} visualization completed. Overall: {mar_at_k:.4f}")
        for c in classes:
            logger.info(f"Class {CIFAR10_CLASSES[c]}: {np.mean(class_recalls[c]):.4f}")
        
        return mar_at_k

@app.command()
def train_ae(
    data_dir: str = "./data",
    latent_dim: int = 128,
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    debug: bool = True,
    visualize_mar: bool = True,
    mar_viz_epochs: int = 5,
    mar_samples_per_class: int = 5,
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
        max_epochs=100,
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
    model = VanillaAutoencoder(
        latent_dim=latent_dim,
        visualize_mar=visualize_mar,
        mar_viz_epochs=mar_viz_epochs,
        mar_samples_per_class=mar_samples_per_class
    )
    
    # Print model architecture and parameter count
    logger.info(f"Model Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Size: {trainable_params:,} trainable parameters ({total_params:,} total)")
    
    if visualize_mar:
        logger.info(f"MAR@5 visualization enabled: Generating visualizations every {mar_viz_epochs} epochs")
        logger.info(f"Using {mar_samples_per_class} samples per class for MAR@5 visualization")
    
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
    Visualize MAR@k for a trained model using CIFAR-10 dataset
    
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
    model = VanillaAutoencoder.load_from_checkpoint(checkpoint_path)
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
        class_names=CIFAR10_CLASSES
    )
    
    logger.info(f"Overall MAR@{k}: {mar_at_k:.4f}")
    logger.info(f"Visualizations saved to: {output_dir}")
    
    return mar_at_k

if __name__ == "__main__":
    app()
