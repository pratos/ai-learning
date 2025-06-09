import os
from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


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
        self.transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        self.transform_val = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        )

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
        # Disable pin_memory when num_workers=0 to avoid issues during profiling
        pin_memory = self.num_workers > 0
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

    def val_dataloader(self):
        # Disable pin_memory when num_workers=0 to avoid issues during profiling
        pin_memory = self.num_workers > 0
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )
