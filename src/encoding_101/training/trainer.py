import os
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger

from src.encoding_101.data import CIFAR10DataModule

ROOT_DIR = Path(__file__).parents[3]


def train_autoencoder(
    model_class,
    model_kwargs: dict = None,
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = os.cpu_count() or 4,
    device_id: int = 0,
    max_epochs: int = 100,
    debug: bool = True,
):
    """
    Train an autoencoder model
    
    Args:
        model_class: The autoencoder class to instantiate
        model_kwargs: Keyword arguments to pass to the model constructor
        data_dir: Directory where the data will be stored
        batch_size: Batch size for training and validation
        num_workers: Number of workers for DataLoader
        device_id: GPU device ID to use
        max_epochs: Maximum number of epochs to train
        debug: Whether to run in debug mode
    
    Returns:
        Trained model and trainer
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ]
    
    # Setup logger
    logger_dir = ROOT_DIR / "logs"
    logger_dir.mkdir(parents=True, exist_ok=True)
    tf_logger = TensorBoardLogger(save_dir=logger_dir)

    logger.info(f"TensorBoard logger active. Saved in directory: {tf_logger.save_dir}")
    
    # Setup trainer
    trainer = Trainer(
        logger=tf_logger,
        accelerator="auto",
        devices=[device_id],
        max_epochs=max_epochs,
        num_nodes=1,
        callbacks=callbacks,
    )
    
    # Setup data module
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
    model = model_class(**model_kwargs)
    
    # Print model architecture and parameter count
    logger.info(f"Model Architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Size: {trainable_params:,} trainable parameters ({total_params:,} total)")
    
    if model_kwargs.get('visualize_mar', False):
        mar_viz_epochs = model_kwargs.get('mar_viz_epochs', 5)
        mar_samples_per_class = model_kwargs.get('mar_samples_per_class', 5)
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
        return model, trainer
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Shutting down gracefully...")
        # Close dataloaders explicitly to avoid worker issues
        if hasattr(training_dataloader, '_iterator'):
            training_dataloader._iterator = None
        if hasattr(validation_dataloader, '_iterator'):
            validation_dataloader._iterator = None
        logger.info("Resources cleaned up. Exiting.")
        return model, trainer 