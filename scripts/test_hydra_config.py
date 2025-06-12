#!/usr/bin/env python3
"""
Test script to verify Hydra configuration system is working correctly
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def test_config(cfg: DictConfig) -> None:
    """Test that the Hydra configuration loads and can instantiate objects correctly"""
    
    logger.info("🧪 Testing Hydra Configuration System")
    logger.info("=" * 50)
    
    # Print the full resolved configuration
    logger.info("📋 Resolved Configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    
    # Test model class loading
    try:
        model_class = hydra.utils.get_class(cfg.model._target_)
        logger.info(f"✅ Model class loaded: {model_class}")
        
        # Test model kwargs
        model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
        model_kwargs.pop('_target_')
        model_kwargs.pop('name')
        logger.info(f"✅ Model kwargs: {model_kwargs}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model class: {e}")
        return
    
    # Test data module instantiation
    try:
        data_module = hydra.utils.instantiate(cfg.data)
        logger.info(f"✅ Data module instantiated: {type(data_module)}")
        logger.info(f"   Data dir: {data_module.data_dir}")
        logger.info(f"   Batch size: {data_module.batch_size}")
        logger.info(f"   Num workers: {data_module.num_workers}")
        
    except Exception as e:
        logger.error(f"❌ Failed to instantiate data module: {e}")
        return
    
    # Test model instantiation
    try:
        model = model_class(**model_kwargs)
        logger.info(f"✅ Model instantiated: {type(model)}")
        logger.info(f"   Latent dim: {model.latent_dim}")
        logger.info(f"   Visualize MAR: {model.visualize_mar}")
        
        # Check if it's an NVTX model
        if hasattr(model, 'enable_nvtx'):
            logger.info(f"   NVTX enabled: {model.enable_nvtx}")
            
    except Exception as e:
        logger.error(f"❌ Failed to instantiate model: {e}")
        return
    
    # Test configuration interpolation
    logger.info("🔍 Testing Configuration Interpolation:")
    logger.info(f"   Experiment name: {cfg.experiment_name}")
    logger.info(f"   Device ID: {cfg.device_id}")
    logger.info(f"   Max epochs: {cfg.training.max_epochs}")
    
    logger.info("🎉 All configuration tests passed!")


if __name__ == "__main__":
    test_config() 