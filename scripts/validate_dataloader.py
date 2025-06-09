#!/usr/bin/env python3
"""
Simple DataLoader validation script to test for segmentation faults
"""
import torch
import typer
from loguru import logger

from src.encoding_101.data import CIFAR10DataModule


def test_dataloader(
    batch_size: int = typer.Argument(256, help="Batch size for testing"),
    num_workers: int = typer.Argument(0, help="Number of DataLoader workers"),
    data_dir: str = typer.Option("./data", help="Directory where the data is stored"),
    max_batches: int = typer.Option(5, help="Maximum number of batches to test"),
):
    """Test DataLoader for segmentation faults before profiling"""
    
    logger.info("🧪 Testing DataLoader for stability...")
    logger.info(f"batch_size={batch_size}, num_workers={num_workers}")
    
    try:
        # Setup data module
        data_module = CIFAR10DataModule(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        
        logger.info("📂 Preparing data...")
        data_module.prepare_data()
        data_module.setup()
        
        # Test training dataloader
        logger.info("🚂 Testing training dataloader...")
        train_loader = data_module.train_dataloader()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            logger.info(f"  Batch {batch_idx}: x.shape={x.shape}, y.shape={y.shape}")
            
            # Test GPU transfer
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                logger.info(f"  GPU transfer successful: {x_gpu.device}")
            
            if batch_idx >= max_batches - 1:
                break
        
        # Test validation dataloader
        logger.info("🔍 Testing validation dataloader...")
        val_loader = data_module.val_dataloader()
        
        for batch_idx, (x, y) in enumerate(val_loader):
            logger.info(f"  Val Batch {batch_idx}: x.shape={x.shape}, y.shape={y.shape}")
            
            if batch_idx >= max_batches - 1:
                break
        
        logger.info("✅ DataLoader test completed successfully!")
        logger.info("🚀 Safe to proceed with profiling")
        
    except Exception as e:
        logger.error(f"❌ DataLoader test failed: {e}")
        logger.error("⚠️  Do not proceed with profiling - fix DataLoader issues first")
        raise


if __name__ == "__main__":
    typer.run(test_dataloader) 