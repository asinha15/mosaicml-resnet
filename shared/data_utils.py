"""
Dataset utilities for ImageNet training with HuggingFace integration.
Supports both full ImageNet and subset loading for testing.
"""
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VisionDataset, ImageNet
import torchvision.transforms as transforms
from datasets import load_dataset
import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image


class ImageNetHF(VisionDataset):
    """
    HuggingFace ImageNet dataset wrapper with subset support.
    Inherits from VisionDataset for Composer compatibility.
    Handles authentication for gated ImageNet-1k dataset.
    """
    
    def __init__(
        self,
        split: str = 'train',
        subset_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        streaming: bool = False,
        token: Optional[str] = None
    ):
        """
        Initialize ImageNet dataset from HuggingFace.
        
        Args:
            split: 'train' or 'validation'
            subset_size: If provided, only load this many samples (useful for testing)
            transform: Torchvision transforms to apply
            streaming: Whether to use streaming (memory efficient for large datasets)
            token: HuggingFace token for accessing gated datasets
        """
        # Initialize VisionDataset parent class
        try:
            super().__init__(root="", transform=None, target_transform=None)
        except TypeError:
            # Fallback for older torchvision versions
            super().__init__(root="")
        
        self.split = split
        self.subset_size = subset_size
        # Set up transforms - use custom one or default
        self.transform = transform or self._get_default_transform()
        self.streaming = streaming
        self.token = token or os.environ.get('HF_TOKEN')
        self._cached_items = {} if streaming else None
        self._iterator = None
        self._current_idx = 0
        
        # Check for offline mode
        offline_mode = os.environ.get('HF_DATASETS_OFFLINE', '0') == '1'
        hf_home = os.environ.get('HF_HOME')
        
        # Check for authentication (only needed for online mode)
        if not offline_mode and not self.token:
            print("⚠️  No HuggingFace token provided. ImageNet-1k is a gated dataset.")
            print("💡 Set HF_TOKEN environment variable or pass token parameter")
        
        # Validate offline mode setup
        if offline_mode and not hf_home:
            print("⚠️  HF_DATASETS_OFFLINE=1 but HF_HOME not set")
            print("💡 Set HF_HOME environment variable to cache directory")
        
        # Load dataset from HuggingFace with error handling
        if offline_mode and hf_home:
            print(f"Loading ImageNet {split} split from LOCAL HuggingFace cache...")
            print(f"   📁 Cache location: {hf_home}")
        else:
            print(f"Loading ImageNet {split} split from HuggingFace (online)...")
        
        try:
            if offline_mode and hf_home:
                # Special handling for offline cached datasets
                print(f"   💾 Loading from offline cache (no Hub connection)")
                
                # Temporarily disable offline mode for loading, then re-enable
                original_offline = os.environ.get('HF_DATASETS_OFFLINE')
                os.environ.pop('HF_DATASETS_OFFLINE', None)
                
                try:
                    if streaming:
                        print(f"   🌊 Using streaming mode with cache")
                        self.dataset = load_dataset(
                            "imagenet-1k", 
                            split=split, 
                            streaming=True,
                            token=self.token,
                            cache_dir=hf_home
                        )
                        
                        if subset_size is not None:
                            self._length = min(subset_size, 1281167 if split == 'train' else 50000)
                            self._subset_size = subset_size
                            print(f"   📊 Will use first {self._length:,} samples from stream")
                        else:
                            self._length = 1281167 if split == 'train' else 50000
                            self._subset_size = None
                    else:
                        print(f"   💾 Loading cached dataset into memory")
                        self.dataset = load_dataset(
                            "imagenet-1k", 
                            split=split, 
                            token=self.token,
                            cache_dir=hf_home
                        )
                        
                        if subset_size is not None:
                            # Sample random subset
                            indices = np.random.choice(len(self.dataset), 
                                                     min(subset_size, len(self.dataset)), 
                                                     replace=False)
                            self.dataset = self.dataset.select(indices)
                            print(f"✅ Created subset with {len(self.dataset)} samples")
                        
                        self._length = len(self.dataset)
                        
                finally:
                    # Restore offline mode
                    if original_offline:
                        os.environ['HF_DATASETS_OFFLINE'] = original_offline
                    
            else:
                # Online mode or no cache
                if streaming:
                    # Use streaming mode - much faster startup
                    print(f"   🌊 Using streaming mode for fast startup")
                    self.dataset = load_dataset(
                        "imagenet-1k", 
                        split=split, 
                        streaming=True,
                        token=self.token
                    )
                    
                    if subset_size is not None:
                        # For streaming with subset, we'll limit during iteration
                        self._length = min(subset_size, 1281167 if split == 'train' else 50000)
                        self._subset_size = subset_size
                        print(f"   📊 Will use first {self._length:,} samples from stream")
                    else:
                        self._length = 1281167 if split == 'train' else 50000
                        self._subset_size = None
                else:
                    # Load subset into memory (slower startup but deterministic sampling)
                    print(f"   💾 Loading dataset into memory for deterministic sampling")
                    self.dataset = load_dataset("imagenet-1k", split=split, token=self.token)
                    
                    if subset_size is not None:
                        # Sample random subset
                        indices = np.random.choice(len(self.dataset), 
                                                 min(subset_size, len(self.dataset)), 
                                                 replace=False)
                        self.dataset = self.dataset.select(indices)
                        print(f"✅ Created subset with {len(self.dataset)} samples")
                    
                    self._length = len(self.dataset)
        
        except Exception as e:
            print(f"❌ Error loading ImageNet: {e}")
            
            if "OfflineModeIsEnabled" in str(e):
                print("\n🔧 OFFLINE MODE ERROR:")
                print("   The HuggingFace library can't load datasets in offline mode.")
                print("   This is expected behavior - the fix is being applied automatically.")
                print("   If this persists, check your cache directory structure.")
            elif "401" in str(e) or "unauthorized" in str(e).lower():
                print("\n🔐 AUTHENTICATION ERROR:")
                print("   ImageNet-1k is a gated dataset requiring access approval.")
                print("   Steps to fix:")
                print("   1. Visit: https://huggingface.co/datasets/imagenet-1k")
                print("   2. Request access (may take 1-2 days)")
                print("   3. Get your token: https://huggingface.co/settings/tokens")
                print("   4. Set token: export HF_TOKEN='your_token' or pass token parameter")
            elif "ConnectionError" in str(e) or "couldn't reach" in str(e).lower():
                print("\n🌐 CONNECTION ERROR:")
                print("   Can't connect to HuggingFace Hub. Possible solutions:")
                print("   1. Check internet connection")
                print("   2. Use cached data with proper environment setup")
                print("   3. Temporarily disable offline mode for initial load")
            else:
                print("💡 Make sure you have access to the imagenet-1k dataset on HuggingFace")
            raise
    
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
        return self._length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.streaming:
            # For streaming with subset, ensure we don't exceed subset bounds
            if hasattr(self, '_subset_size') and self._subset_size is not None:
                if idx >= self._subset_size:
                    raise IndexError(f"Index {idx} exceeds subset size {self._subset_size}")
            
            # Pre-fetch and cache items for better performance
            if not hasattr(self, '_cached_items') or self._cached_items is None:
                self._cached_items = {}
                self._iterator = iter(self.dataset)
                self._current_idx = 0
                self._cache_hits = 0
                self._cache_misses = 0
                print(f"   🔧 Initialized streaming cache for dataset")
            
            # Debug info every 1000 requests
            total_requests = getattr(self, '_cache_hits', 0) + getattr(self, '_cache_misses', 0)
            if total_requests > 0 and total_requests % 1000 == 0:
                hit_rate = self._cache_hits / total_requests * 100
                print(f"   📊 Cache stats: {hit_rate:.1f}% hit rate, {len(self._cached_items)} items cached")
            
            # Check if we need to restart the iterator (new epoch or large backward jump)
            if idx < self._current_idx and idx not in self._cached_items:
                # DataLoader is asking for an earlier index - probably new epoch
                # Need to restart the stream iterator
                jump_size = self._current_idx - idx
                print(f"   🔄 Epoch restart detected: requesting idx {idx}, current iterator at {self._current_idx} (backward jump: {jump_size})")
                self._iterator = iter(self.dataset)
                self._current_idx = 0
                old_cache_size = len(self._cached_items)
                self._cached_items.clear()  # Clear cache since we're restarting
                print(f"   🔄 Stream iterator restarted, cache cleared ({old_cache_size} items removed)")
                
                # Reset cache statistics for new epoch
                if not hasattr(self, '_epoch_count'):
                    self._epoch_count = 1
                else:
                    self._epoch_count += 1
                print(f"   📊 Starting epoch {self._epoch_count}")
            
            # Return cached item if available
            if idx in self._cached_items:
                item = self._cached_items[idx]
                self._cache_hits += 1
            else:
                self._cache_misses += 1
                
                # Check if requested index is reasonable
                if idx < self._current_idx - 1000:
                    # Asking for an index way behind current position - something's wrong
                    print(f"   ⚠️  Warning: Large backward jump detected (idx {idx}, current {self._current_idx})")
                
                # Sequential access pattern - fetch items until we reach the desired index
                while self._current_idx <= idx:
                    try:
                        item_data = next(self._iterator)
                        self._cached_items[self._current_idx] = item_data
                        
                        # Store the item we need before potentially removing it from cache
                        if self._current_idx == idx:
                            item = item_data
                        
                        self._current_idx += 1
                        
                        # Limit cache size to prevent memory issues
                        # Keep reasonable cache size but preserve recently accessed items
                        if len(self._cached_items) > 500:  # Reduced cache size for better management
                            # Keep the most recent 200 items
                            all_keys = sorted(self._cached_items.keys())
                            keys_to_keep = all_keys[-200:]  # Keep last 200 items
                            
                            # Create new cache with only recent items
                            new_cache = {k: self._cached_items[k] for k in keys_to_keep if k in self._cached_items}
                            self._cached_items = new_cache
                            
                            # Debug info
                            if len(new_cache) < len(all_keys):
                                print(f"   🧹 Cache cleanup: {len(all_keys)} → {len(new_cache)} items")
                                    
                    except StopIteration:
                        # End of dataset reached
                        if idx >= self._length:
                            raise IndexError(f"Index {idx} exceeds dataset length {self._length}")
                        else:
                            # Premature end - restart iterator and try again
                            print(f"   🔄 Iterator exhausted prematurely, restarting...")
                            self._iterator = iter(self.dataset)
                            self._current_idx = 0
                            self._cached_items.clear()
                            continue  # Try again with fresh iterator
                
                # Ensure we have the item
                if idx not in self._cached_items:
                    # This shouldn't happen with the new logic, but provide good error info
                    cached_keys = sorted(self._cached_items.keys()) if self._cached_items else []
                    cache_range = f"[{min(cached_keys)}...{max(cached_keys)}]" if cached_keys else "[]"
                    
                    raise RuntimeError(
                        f"DATASET ACCESS ERROR: Cannot access index {idx}. "
                        f"Current iterator position: {self._current_idx}, "
                        f"Cache has {len(self._cached_items)} items in range {cache_range}. "
                        f"This suggests a DataLoader configuration issue with streaming datasets."
                    )
                
                item = self._cached_items[idx]
        else:
            item = self.dataset[idx]
        
        # Process image
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Ensure image is RGB (convert grayscale to RGB if needed)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
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
    use_hf: bool = True,
    streaming: bool = False,
    token: Optional[str] = None
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
            streaming=streaming,
            token=token
        )
        
        val_dataset = ImageNetHF(
            split='validation',
            subset_size=min(subset_size // 10, 5000) if subset_size else None,
            transform=val_transforms,
            streaming=streaming,  # Use same streaming setting as training
            token=token
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
    
    # Create dataloaders with streaming optimizations
    # For streaming datasets, disable shuffling and reduce workers to avoid random access issues
    if use_hf and streaming:
        use_shuffle = False  # Don't shuffle streaming datasets
        actual_workers = 0  # Streaming works better with single-threaded loading
        print(f"   🔧 Applying streaming optimizations: shuffle={use_shuffle}, workers={actual_workers}")
    else:
        use_shuffle = True
        actual_workers = num_workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        persistent_workers=actual_workers > 0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        persistent_workers=actual_workers > 0,
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
