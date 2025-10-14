"""
Dataset utilities for ImageNet training with HuggingFace integration.
Supports both full ImageNet and subset loading for testing.
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from datasets import load_dataset
import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image


class ImageNetHF:
    """
    HuggingFace ImageNet dataset wrapper with subset support.
    """
    
    def __init__(
        self,
        split: str = 'train',
        subset_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        streaming: bool = False
    ):
        """
        Initialize ImageNet dataset from HuggingFace.
        
        Args:
            split: 'train' or 'validation'
            subset_size: If provided, only load this many samples (useful for testing)
            transform: Torchvision transforms to apply
            streaming: Whether to use streaming (memory efficient for large datasets)
        """
        self.split = split
        self.subset_size = subset_size
        self.transform = transform or self._get_default_transform()
        self.streaming = streaming
        
        # Load dataset from HuggingFace with error handling
        print(f"Loading ImageNet {split} split from HuggingFace...")
        try:
            if streaming and subset_size is None:
                # Use streaming for full dataset
                self.dataset = load_dataset(
                    "imagenet-1k", 
                    split=split, 
                    streaming=True
                )
            else:
                # Load into memory (required for subsetting)
                self.dataset = load_dataset(
                    "imagenet-1k", 
                    split=split
                )
                
                # Create subset if specified
                if subset_size is not None:
                    indices = np.random.choice(len(self.dataset), subset_size, replace=False)
                    self.dataset = self.dataset.select(indices)
                    print(f"Created subset with {len(self.dataset)} samples")
        
        except Exception as e:
            print(f"Warning: Failed to load ImageNet from HuggingFace: {e}")
            print("For testing purposes, consider using CIFAR-10 or a local dataset.")
            raise RuntimeError(
                f"ImageNet loading failed: {e}. "
                "Please ensure you have access to the imagenet-1k dataset on HuggingFace, "
                "or use a local dataset with --use-hf=False"
            )
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default transforms based on split."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        if self.streaming:
            # Streaming datasets don't have len, estimate based on split
            return 1281167 if self.split == 'train' else 50000
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.streaming:
            # For streaming, we need to iterate
            item = next(iter(self.dataset.skip(idx).take(1)))
        else:
            item = self.dataset[idx]
        
        # Convert PIL image to tensor
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = item['label']
        return image, label


def get_imagenet_transforms(
    image_size: int = 224,
    augment_train: bool = True,
    mixup_cutmix: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get ImageNet transforms optimized for Composer training.
    
    Args:
        image_size: Target image size
        augment_train: Whether to use augmentation for training
        mixup_cutmix: Whether transforms should be compatible with MixUp/CutMix
        
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    
    # Validation transforms (always the same)
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # Slightly larger for crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Training transforms
    if augment_train:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                ratio=(3./4., 4./3.)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transforms = val_transforms
    
    return train_transforms, val_transforms


def create_dataloaders(
    batch_size: int = 256,
    num_workers: int = 8,
    subset_size: Optional[int] = None,
    image_size: int = 224,
    pin_memory: bool = True,
    use_hf: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        subset_size: If provided, use only this many samples (for testing)
        image_size: Target image size
        pin_memory: Whether to pin memory for faster GPU transfer
        use_hf: Whether to use HuggingFace dataset (True) or local ImageNet (False)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    train_transforms, val_transforms = get_imagenet_transforms(image_size)
    
    if use_hf:
        # Use HuggingFace ImageNet
        print("Using HuggingFace ImageNet dataset...")
        
        train_dataset = ImageNetHF(
            split='train',
            subset_size=subset_size,
            transform=train_transforms,
            streaming=subset_size is None and subset_size != 'test'
        )
        
        val_dataset = ImageNetHF(
            split='validation',
            subset_size=min(subset_size // 10, 5000) if subset_size else None,
            transform=val_transforms,
            streaming=False
        )
    else:
        # Use local ImageNet dataset (requires manual download)
        imagenet_root = os.environ.get('IMAGENET_ROOT', './data/imagenet')
        if not os.path.exists(imagenet_root):
            raise ValueError(
                f"ImageNet root {imagenet_root} not found. "
                "Either set IMAGENET_ROOT environment variable or use use_hf=True"
            )
        
        train_dataset = ImageNet(
            root=imagenet_root,
            split='train',
            transform=train_transforms
        )
        
        val_dataset = ImageNet(
            root=imagenet_root,
            split='val',
            transform=val_transforms
        )
        
        # Create subsets if requested
        if subset_size:
            train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
            val_indices = np.random.choice(len(val_dataset), min(subset_size // 10, 5000), replace=False)
            
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False
    )
    
    print(f"Created dataloaders:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return train_loader, val_loader


# Test subset sizes for different scenarios
SUBSET_SIZES = {
    'tiny': 1000,      # Very quick test
    'small': 10000,    # Small test
    'medium': 100000,  # Medium test
    'large': 500000,   # Large test
    'full': None       # Full dataset
}


def get_subset_size(subset_name: str) -> Optional[int]:
    """Get subset size by name."""
    return SUBSET_SIZES.get(subset_name, subset_name if isinstance(subset_name, int) else None)
