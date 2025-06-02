import io
from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch import LightningModule
from loguru import logger

from src.encoding_101.metrics import compute_mar_at_k
from src.encoding_101.utils import create_comparison_grid

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class BaseAutoencoder(LightningModule, ABC):
    """Base class for all autoencoder models"""
    
    def __init__(
        self, 
        latent_dim: int = 128, 
        visualize_mar: bool = False, 
        mar_viz_epochs: int = 5, 
        mar_samples_per_class: int = 5
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.save_hyperparameters()
        
        # MAR visualization settings
        self.visualize_mar = visualize_mar
        self.mar_viz_epochs = mar_viz_epochs
        self.mar_samples_per_class = mar_samples_per_class
        
        # Store images for visualization at the end of each epoch
        self.train_imgs = []
        self.train_recon_imgs = []
        self.val_imgs = []
        self.val_recon_imgs = []
        
        # For MAR@5 calculation
        self.val_embeddings = []
        self.val_labels = []
        self.val_images = []
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input images"""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to images"""
        pass
    
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
            # Use the utility function to create comparison grid
            image_tensor = create_comparison_grid(
                self.train_imgs,
                self.train_recon_imgs, 
                "Training",
                self.current_epoch
            )
            
            if image_tensor is not None:
                # Log to TensorBoard with consistent tag name for slider effect
                self.logger.experiment.add_image("train_comparison", image_tensor, self.current_epoch)
    
    def on_validation_epoch_end(self):
        """Log validation images and compute MAR@5 at the end of each validation epoch"""
        # Process validation images
        if len(self.val_imgs) > 0:
            # Use the utility function to create comparison grid
            image_tensor = create_comparison_grid(
                self.val_imgs,
                self.val_recon_imgs,
                "Validation", 
                self.current_epoch
            )
            
            if image_tensor is not None:
                # Log to TensorBoard with consistent tag name for slider effect
                self.logger.experiment.add_image("val_comparison", image_tensor, self.current_epoch)
        
        # Compute MAR@5 if we have embeddings
        if self.val_embeddings:
            # Concatenate all stored embeddings and labels
            all_embeddings = torch.cat(self.val_embeddings, dim=0)
            all_labels = torch.cat(self.val_labels, dim=0)
            
            # Compute MAR@5 using the imported function
            mar_at_5 = compute_mar_at_k(all_embeddings, all_labels, k=5)
            
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