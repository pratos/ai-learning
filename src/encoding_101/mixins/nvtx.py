from contextlib import nullcontext

import nvtx
import torch
import torch.nn.functional as F
from loguru import logger

from src.encoding_101.metrics import compute_mar_at_k
from src.encoding_101.utils import create_comparison_grid


class NVTXColors:
    """Centralized color scheme for NVTX annotations"""
    TRAIN = "green"
    VALIDATION = "blue"
    FORWARD = "yellow"
    LOSS = "orange"
    BACKWARD = "red"
    METRICS = "purple"
    VISUALIZATION = "pink"
    DATA_TRANSFER = "cyan"
    MEMORY = "magenta"
    # Epoch-level markers
    EPOCH_START = "lime"
    EPOCH_END = "darkgreen"


class NVTXProfilingMixin:
    """Mixin class that adds NVTX profiling capabilities to any autoencoder"""
    
    def __init__(self, *args, enable_nvtx: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_nvtx = enable_nvtx
        self._epoch_nvtx_context = None
    
    def nvtx_annotate(self, name: str, color: str = "white"):
        """Conditional NVTX annotation context manager"""
        if self.enable_nvtx:
            return nvtx.annotate(name, color=color)
        else:
            return nullcontext()
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input images with NVTX annotations"""
        with self.nvtx_annotate("Encoder Forward", NVTXColors.FORWARD):
            return super().encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode embeddings to images with NVTX annotations"""
        with self.nvtx_annotate("Decoder Forward", NVTXColors.FORWARD):
            return super().decode(z)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        with self.nvtx_annotate(f"Training Step {batch_idx}", NVTXColors.TRAIN):
            
            with self.nvtx_annotate("Data Unpack", NVTXColors.DATA_TRANSFER):
                x, y = batch
            
            with self.nvtx_annotate("Forward Pass", NVTXColors.FORWARD):
                x_hat = self(x)
            
            with self.nvtx_annotate("Loss Computation", NVTXColors.LOSS):
                loss = F.mse_loss(x_hat, x)
                self.log("train_loss", loss)
            
            # Store images from first batch for visualization
            if batch_idx == 0:
                with self.nvtx_annotate("Store Train Images", NVTXColors.VISUALIZATION):
                    n_images = min(8, x.size(0))
                    self.train_imgs = x[:n_images].detach().clone()
                    self.train_recon_imgs = x_hat[:n_images].detach().clone()
            
            return loss
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        with self.nvtx_annotate(f"Validation Step {batch_idx}", NVTXColors.VALIDATION):
            
            with self.nvtx_annotate("Val Data Unpack", NVTXColors.DATA_TRANSFER):
                x, y = batch
            
            with self.nvtx_annotate("Val Forward Pass", NVTXColors.FORWARD):
                x_hat = self(x)
            
            with self.nvtx_annotate("Val Loss Computation", NVTXColors.LOSS):
                loss = F.mse_loss(x_hat, x)
                self.log("val_loss", loss)
            
            # Store the first batch for visualization
            if batch_idx == 0:
                with self.nvtx_annotate("Store Val Images", NVTXColors.VISUALIZATION):
                    n_images = min(8, x.size(0))
                    self.val_imgs = x[:n_images].detach().clone()
                    self.val_recon_imgs = x_hat[:n_images].detach().clone()
            
            # Store embeddings, labels and images for MAR@5 calculation
            with self.nvtx_annotate("Collect Embeddings", NVTXColors.METRICS):
                embeddings = self.encode(x)
                self.val_embeddings.append(embeddings.detach())
                self.val_labels.append(y.detach())
                
                # Only store images if we're going to visualize MAR
                if self.visualize_mar and self.current_epoch % self.mar_viz_epochs == 0:
                    self.val_images.append(x.detach().cpu())
            
            return loss
    
    def on_train_epoch_start(self):
        """Mark the start of training epoch with NVTX annotation"""
        # Create epoch boundary markers
        with self.nvtx_annotate(f"🚀 EPOCH {self.current_epoch} - TRAINING START", NVTXColors.EPOCH_START):
            pass
        
        # Start epoch-wide range (will be closed in on_validation_epoch_end)
        if self.enable_nvtx:
            self._epoch_nvtx_context = nvtx.annotate(f"EPOCH {self.current_epoch} FULL CYCLE", color="white")
            self._epoch_nvtx_context.__enter__()
        
        # Call parent method if it exists
        if hasattr(super(), 'on_train_epoch_start'):
            super().on_train_epoch_start()
    
    def on_train_epoch_end(self):
        """Log training images with NVTX annotations"""
        with self.nvtx_annotate(f"✅ EPOCH {self.current_epoch} - TRAINING END", NVTXColors.EPOCH_END):
            pass
        
        with self.nvtx_annotate(f"Train Epoch {self.current_epoch} Visualization", NVTXColors.VISUALIZATION):
            if len(self.train_imgs) > 0:
                with self.nvtx_annotate("Create Train Comparison Grid", NVTXColors.VISUALIZATION):
                    image_tensor = create_comparison_grid(
                        self.train_imgs,
                        self.train_recon_imgs, 
                        "Training",
                        self.current_epoch
                    )
                    
                    if image_tensor is not None:
                        with self.nvtx_annotate("Log Train Images", NVTXColors.VISUALIZATION):
                            self.logger.experiment.add_image("train_comparison", image_tensor, self.current_epoch)
        
        # Call parent method if it exists
        if hasattr(super(), 'on_train_epoch_end'):
            super().on_train_epoch_end()
    
    def on_validation_epoch_start(self):
        """Mark the start of validation epoch with NVTX annotation"""
        with self.nvtx_annotate(f"🔍 EPOCH {self.current_epoch} - VALIDATION START", NVTXColors.EPOCH_START):
            pass
        
        # Call parent method if it exists
        if hasattr(super(), 'on_validation_epoch_start'):
            super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self):
        """Enhanced validation epoch end with comprehensive NVTX annotations"""
        with self.nvtx_annotate(f"✅ EPOCH {self.current_epoch} - VALIDATION END", NVTXColors.EPOCH_END):
            pass
        
        with self.nvtx_annotate(f"Validation Epoch {self.current_epoch} Metrics", NVTXColors.METRICS):
            
            # Process validation images
            if len(self.val_imgs) > 0:
                with self.nvtx_annotate("Create Val Comparison Grid", NVTXColors.VISUALIZATION):
                    image_tensor = create_comparison_grid(
                        self.val_imgs,
                        self.val_recon_imgs,
                        "Validation", 
                        self.current_epoch
                    )
                    
                    if image_tensor is not None:
                        with self.nvtx_annotate("Log Val Images", NVTXColors.VISUALIZATION):
                            self.logger.experiment.add_image("val_comparison", image_tensor, self.current_epoch)
            
            # Compute MAR@5 if we have embeddings
            if self.val_embeddings:
                with self.nvtx_annotate("MAR@5 Computation", NVTXColors.METRICS):
                    # Concatenate all stored embeddings and labels
                    all_embeddings = torch.cat(self.val_embeddings, dim=0)
                    all_labels = torch.cat(self.val_labels, dim=0)
                    
                    # Compute MAR@5
                    mar_at_5 = compute_mar_at_k(all_embeddings, all_labels, k=5)
                    
                    # Log the metric
                    self.log("val_mar_at_5", mar_at_5)
                    logger.info(f"Epoch {self.current_epoch}: MAR@5 = {mar_at_5:.4f}")
                
                # Visualize MAR@5 periodically during training if enabled
                if (self.visualize_mar and 
                    self.current_epoch % self.mar_viz_epochs == 0 and 
                    len(self.val_images) > 0):
                    
                    with self.nvtx_annotate("MAR@5 Visualization", NVTXColors.VISUALIZATION):
                        all_images = torch.cat(self.val_images, dim=0)
                        
                        # Get the log directory for this run
                        tensorboard_dir = self.logger.log_dir if hasattr(self.logger, 'log_dir') else None
                        if tensorboard_dir:
                            from pathlib import Path
                            mar_viz_dir = Path(tensorboard_dir) / f"mar_viz_epoch_{self.current_epoch}"
                            mar_viz_dir.mkdir(parents=True, exist_ok=True)
                            
                            logger.info(f"Visualizing MAR@5 for epoch {self.current_epoch}...")
                            
                            # Prepare the data for visualization
                            with self.nvtx_annotate("Prepare MAR Data", NVTXColors.METRICS):
                                class_indices = {}
                                for i in range(10):  # CIFAR-10 classes
                                    class_indices[i] = (all_labels == i).nonzero(as_tuple=True)[0]
                                
                                # Select samples for each class
                                selected_indices = []
                                for class_idx in range(10):
                                    indices = class_indices[class_idx]
                                    if len(indices) > self.mar_samples_per_class:
                                        selected = indices[torch.randperm(len(indices))[:self.mar_samples_per_class]]
                                        selected_indices.append(selected)
                                    else:
                                        selected_indices.append(indices)
                                
                                # Flatten the list
                                if selected_indices:
                                    selected_indices = torch.cat(selected_indices)
                            
                            # Generate visualizations
                            if len(selected_indices) > 0:
                                with self.nvtx_annotate("Generate MAR Visualizations", NVTXColors.VISUALIZATION):
                                    self.visualize_mar_for_selected(
                                        all_images, all_embeddings, all_labels, 
                                        selected_indices, mar_viz_dir, k=5
                                    )
                                    
                                    # Log a summary figure to TensorBoard
                                    summary_path = mar_viz_dir / "mar_at_5_summary.png"
                                    if summary_path.exists():
                                        import PIL
                                        import torchvision
                                        summary_img = PIL.Image.open(summary_path)
                                        summary_tensor = torchvision.transforms.ToTensor()(summary_img)
                                        self.logger.experiment.add_image(
                                            "MAR@5_summary", summary_tensor, self.current_epoch
                                        )
                
                # Clear for next epoch
                self.val_embeddings = []
                self.val_labels = []
                self.val_images = []
        
        # Close epoch-wide NVTX range
        if self.enable_nvtx and self._epoch_nvtx_context is not None:
            self._epoch_nvtx_context.__exit__(None, None, None)
            self._epoch_nvtx_context = None
        
        # Call parent method if it exists
        if hasattr(super(), 'on_validation_epoch_end'):
            super().on_validation_epoch_end()
    
    def configure_optimizers(self):
        """Configure optimizer with NVTX annotation for initialization"""
        with self.nvtx_annotate("Configure Optimizer", NVTXColors.METRICS):
            return super().configure_optimizers()



class NVTXProfiler:
    """Utility class for advanced NVTX profiling patterns"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def annotate(self, name: str, color: str = "white"):
        """Conditional NVTX annotation"""
        if self.enabled:
            return nvtx.annotate(name, color=color)
        else:
            return nullcontext()
    
    def profile_memory(self, operation_name: str, color: str = NVTXColors.MEMORY):
        """Context manager that profiles GPU memory usage"""
        def memory_profiler():
            if torch.cuda.is_available() and self.enabled:
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
                
                with nvtx.annotate(f"{operation_name} [GPU Memory]", color=color):
                    yield
                
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated()
                mem_diff = mem_after - mem_before
                
                logger.debug(f"{operation_name}: Memory change = {mem_diff / 1024**2:.2f} MB")
            else:
                with self.annotate(operation_name, color):
                    yield
        
        return memory_profiler()
    
    def profile_dataloader(self, dataloader, max_batches: int = None, phase: str = "train"):
        """Profile dataloader performance"""
        color = NVTXColors.TRAIN if phase == "train" else NVTXColors.VALIDATION
        
        with self.annotate(f"DataLoader Profile - {phase.title()}", color):
            for batch_idx, batch in enumerate(dataloader):
                with self.annotate(f"Load Batch {batch_idx}", NVTXColors.DATA_TRANSFER):
                    # The batch is loaded here
                    pass
                
                if max_batches and batch_idx >= max_batches:
                    break
                    
                yield batch_idx, batch 