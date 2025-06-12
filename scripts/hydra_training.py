#!/usr/bin/env python3
"""
Hydra-based training script for autoencoder experiments
Replaces the typer-based approach with hierarchical configuration management
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from pathlib import Path
import sys

from src.encoding_101.training.trainer import train_autoencoder


def setup_loguru_logging():
    """Configure pure loguru logging to work with Hydra"""
    # Get Hydra's working directory (where logs should go)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    
    # Configure loguru to write to Hydra's log file
    log_file = output_dir / f"{hydra_cfg.job.name}.log"
    
    # Remove default loguru handler and add our custom ones
    logger.remove()
    
    # Add console handler with beautiful colors and formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    # Add file handler with detailed formatting
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        compression="zip",
        enqueue=True  # Thread-safe logging
    )
    
    logger.info(f"📝 Loguru logging configured - logs will be saved to: {log_file}")
    return log_file


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Train autoencoder with Hydra configuration management
    
    Examples:
        # Default training
        python scripts/hydra_training.py
        
        # Override model and training params
        python scripts/hydra_training.py model=cnn_autoencoder training.max_epochs=20
        
        # Profiling run
        python scripts/hydra_training.py training=profiling
        
        # Multi-run experiments
        python scripts/hydra_training.py model=nvtx_vanilla_autoencoder,cnn_autoencoder training.max_epochs=5,10,20 --multirun
    """
    
    # Setup pure loguru logging
    log_file = setup_loguru_logging()
    
    # Print resolved configuration
    logger.info("🔮 Hydra Configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    # Instantiate model class
    logger.info("🏗️ Instantiating model...")
    model_class = hydra.utils.get_class(cfg.model._target_)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop('_target_')  # Remove hydra metadata
    model_kwargs.pop('name')      # Remove our custom name field
    logger.success(f"✅ Model class loaded: {model_class.__name__}")
    
    # Instantiate data module using Hydra
    logger.info("📊 Instantiating data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    logger.success(f"✅ Data module loaded: {type(data_module).__name__}")
    
    # Train the model with the pre-instantiated data module
    logger.info("🚀 Starting training process...")
    model, trainer = train_autoencoder(
        model_class=model_class,
        model_kwargs=model_kwargs,
        data_module=data_module,  # Use Hydra-instantiated data module
        device_id=cfg.device_id,
        max_epochs=cfg.training.max_epochs,
        debug=cfg.debug,
        experiment_name=cfg.experiment_name,
    )
    
    logger.success("✅ Training complete!")
    logger.info(f"📊 Experiment: {cfg.experiment_name}")
    logger.info(f"📁 Model saved in: {trainer.logger.log_dir}")
    logger.info(f"🎯 TensorBoard: tensorboard --logdir={trainer.logger.log_dir}")
    logger.info(f"📝 Full logs saved to: {log_file}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def profile(cfg: DictConfig) -> None:
    """
    Profile training with NVTX annotations
    
    Examples:
        python scripts/hydra_training.py profile
        python scripts/hydra_training.py profile training=profiling
    """
    
    # Setup pure loguru logging
    setup_loguru_logging()
    
    # Force profiling mode overrides
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys
    cfg.training.max_epochs = 3
    cfg.model.visualize_mar = False
    cfg.profiling.enable_nvtx = True
    OmegaConf.set_struct(cfg, True)   # Re-enable struct mode
    
    logger.info("🔍 NVTX Training Profiler")
    logger.info("=" * 50)
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Experiment: {cfg.experiment_name}")
    logger.info(f"NVTX enabled: {cfg.profiling.enable_nvtx}")
    logger.info(f"Epochs: {cfg.training.max_epochs}")
    logger.info(f"Batch size: {cfg.training.batch_size}")
    logger.info("=" * 50)
    
    # Call training function (but skip setup_logging since we already did it)
    # Print resolved configuration
    logger.info("🔮 Hydra Configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    # Instantiate model class
    logger.info("🏗️ Instantiating model...")
    model_class = hydra.utils.get_class(cfg.model._target_)
    model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
    model_kwargs.pop('_target_')  # Remove hydra metadata
    model_kwargs.pop('name')      # Remove our custom name field
    logger.success(f"✅ Model class loaded: {model_class.__name__}")
    
    # Instantiate data module using Hydra
    logger.info("📊 Instantiating data module...")
    data_module = hydra.utils.instantiate(cfg.data)
    logger.success(f"✅ Data module loaded: {type(data_module).__name__}")
    
    # Train the model with the pre-instantiated data module
    logger.info("🚀 Starting profiling run...")
    model, trainer = train_autoencoder(
        model_class=model_class,
        model_kwargs=model_kwargs,
        data_module=data_module,  # Use Hydra-instantiated data module
        device_id=cfg.device_id,
        max_epochs=cfg.training.max_epochs,
        debug=cfg.debug,
        experiment_name=cfg.experiment_name,
    )
    
    logger.success("✅ Profiling complete!")


if __name__ == "__main__":
    # Simple routing based on command line args
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        sys.argv.pop(1)  # Remove the command from args
        profile()
    else:
        train() 