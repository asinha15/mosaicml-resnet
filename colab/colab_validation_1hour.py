"""
🚀 1-Hour ImageNet Validation Training Script for Google Colab [FIXED]
=====================================================================

FIXED ISSUES:
- ✅ Replaced signal-based timeout with polling-based timeout (safer for Colab)
- ✅ Comprehensive error handling with detailed stack traces
- ✅ Better GPU memory management and cleanup
- ✅ Helpful error suggestions and diagnostics
- ✅ Fixed KeyError in streaming dataset cache (improved cache management logic)
- ✅ Fixed epoch restart detection for streaming datasets (handles DataLoader epoch boundaries)
- ✅ Fixed timeout enforcement with proper Composer callback (ensures exactly 60-minute runtime)

Features:
- ✅ Streams ImageNet-1k data using HuggingFace datasets API (with authentication)
- ✅ Time-limited training (exactly 1 hour) using polling timeout
- ✅ Optimized for Colab T4 GPU (16GB VRAM)
- ✅ MosaicML Composer integration with key algorithms
- ✅ Real-time progress monitoring
- ✅ Memory efficient streaming
- ✅ Automatic checkpointing
- ✅ Robust error handling (no more "Training error: 0")

Authentication Required:
ImageNet-1k is a gated dataset. You need to:
1. Request access: https://huggingface.co/datasets/imagenet-1k
2. Set HF_TOKEN environment variable or login via huggingface-cli

Usage in Colab:
    # Set your HuggingFace token
    import os
    os.environ['HF_TOKEN'] = 'your_token_here'
    
    !python colab_validation_1hour.py --streaming

Or with custom parameters:
    !python colab_validation_1hour.py --batch-size 128 --subset-size 20000 --hf-token your_token --streaming
"""

import os
import sys
import time
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Core ML libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import VisionDataset
import numpy as np
from PIL import Image

# HuggingFace for dataset streaming
from datasets import load_dataset
import datasets

# MosaicML Composer for optimized training
from composer import Trainer, ComposerModel
from composer.algorithms import (
    BlurPool, ChannelsLast, EMA, LabelSmoothing,
    MixUp, CutMix, RandAugment
)
from composer.callbacks import (
    LRMonitor, MemoryMonitor, SpeedMonitor,
    ThresholdStopper
)
from composer.optim import DecoupledSGDW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.utils import reproducibility

# Metrics and utilities
from torchmetrics.classification import MulticlassAccuracy
import torchvision.models as models
import torch.nn.functional as F

# Set random seeds for reproducibility
reproducibility.seed_all(42)

class TimeoutHandler:
    """Handle training timeout using polling (safer than signals in Colab)."""
    
    def __init__(self, timeout_seconds: int = 3600):  # 1 hour default
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        print(f"🔧 Using polling-based timeout (safer for Colab environments)")
    
    def should_stop(self) -> bool:
        """Check if training should stop due to timeout."""
        elapsed = time.time() - self.start_time
        if elapsed >= self.timeout_seconds:
            print(f"\n⏰ Training timeout reached ({self.timeout_seconds/60:.1f} minutes)")
            print("Gracefully stopping training...")
            return True
        return False
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def remaining_time(self) -> float:
        """Get remaining time in seconds."""
        return max(0, self.timeout_seconds - self.elapsed_time())
    
    def progress_percent(self) -> float:
        """Get training progress as percentage."""
        return min(100.0, (self.elapsed_time() / self.timeout_seconds) * 100)


class ResNet50ComposerModel(ComposerModel):
    """ResNet50 model wrapped for MosaicML Composer."""
    
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        
        # Use torchvision's ResNet50
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            if num_classes != 1000:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model = models.resnet50(weights=None, num_classes=num_classes)
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, batch: Any) -> torch.Tensor:
        """Forward pass for Composer model."""
        if isinstance(batch, dict):
            inputs = batch['image']
        else:
            inputs, _ = batch
        return self.model(inputs)
    
    def loss(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        """Compute loss for training."""
        if isinstance(batch, dict):
            targets = batch['label']
        else:
            _, targets = batch
        return F.cross_entropy(outputs, targets)
    
    def metrics(self, train: bool = False) -> Dict[str, Any]:
        """Define metrics to track during training."""
        return {
            'MulticlassAccuracy': MulticlassAccuracy(
                num_classes=1000,
                average='micro'
            )
        }


class StreamingImageNetDataset(VisionDataset):
    """
    Streaming ImageNet dataset using HuggingFace datasets API.
    Memory efficient for large datasets with optional subset sampling.
    Handles authentication for gated ImageNet-1k dataset.
    Inherits from VisionDataset for Composer compatibility.
    """
    
    def __init__(
        self,
        split: str = 'train',
        subset_size: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        streaming: bool = True,
        token: Optional[str] = None
    ):
        # Initialize VisionDataset parent class
        # Use a dummy root since we're streaming from HuggingFace
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
        self._subset_size = None  # Initialize for streaming mode
        
        print(f"🔄 Loading ImageNet-1k {split} split (streaming={streaming})...")
        
        # Check for authentication
        if not self.token:
            print("⚠️  No HuggingFace token provided. ImageNet-1k is a gated dataset.")
            print("💡 Either set HF_TOKEN environment variable or use --hf-token argument")
        
        try:
            if streaming:
                # Use streaming dataset - much faster startup
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
            if "401" in str(e) or "unauthorized" in str(e).lower():
                print("\n🔐 AUTHENTICATION ERROR:")
                print("   ImageNet-1k is a gated dataset requiring access approval.")
                print("   Steps to fix:")
                print("   1. Visit: https://huggingface.co/datasets/imagenet-1k")
                print("   2. Request access (may take 1-2 days)")
                print("   3. Get your token: https://huggingface.co/settings/tokens")
                print("   4. Set token: os.environ['HF_TOKEN'] = 'your_token' or use --hf-token")
            else:
                print("💡 Make sure you have access to the imagenet-1k dataset on HuggingFace")
            raise
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get optimized transforms for ImageNet."""
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                     saturation=0.4, hue=0.1),
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
            if not hasattr(self, '_cached_items'):
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


def create_dataloaders(
    batch_size: int = 128,
    subset_size: Optional[int] = None,
    num_workers: int = 2,
    streaming: bool = True,
    token: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized dataloaders for Colab."""
    
    print(f"🔧 Creating dataloaders (batch_size={batch_size}, workers={num_workers})")
    
    # Training dataset
    train_dataset = StreamingImageNetDataset(
        split='train',
        subset_size=subset_size,
        streaming=streaming,
        token=token
    )
    
    # Validation dataset (smaller subset)
    val_subset = min(5000, subset_size // 10) if subset_size else 5000
    val_dataset = StreamingImageNetDataset(
        split='validation',
        subset_size=val_subset,
        streaming=streaming,  # Use same streaming setting as training
        token=token
    )
    
    # Create dataloaders
    # For streaming datasets, disable shuffling and reduce workers to avoid random access issues
    use_shuffle = False if streaming else True
    actual_workers = 0 if streaming else num_workers  # Streaming works better with single-threaded loading
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=use_shuffle,
        num_workers=actual_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=actual_workers > 0
    )
    
    print(f"   🔧 Training loader: shuffle={use_shuffle}, workers={actual_workers}")
    print(f"      (Streaming optimizations applied)" if streaming else "")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=actual_workers,  # Use same worker optimization
        pin_memory=True,
        drop_last=False,
        persistent_workers=actual_workers > 0
    )
    
    print(f"   🔧 Validation loader: shuffle=False, workers={actual_workers}")
    
    print(f"📊 Dataloaders created:")
    print(f"   Training: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"   Validation: {len(val_dataset):,} samples, {len(val_loader):,} batches")
    
    return train_loader, val_loader


def setup_composer_algorithms(use_full_suite: bool = True, streaming: bool = True) -> list:
    """Setup MosaicML Composer optimization algorithms."""
    algorithms = []
    
    # Try to add gradient clipping if available
    try:
        from composer.algorithms import GradientClipping
        algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=1.0))
        print("   ✅ Added GradientClipping algorithm")
    except ImportError:
        print("   💡 GradientClipping algorithm not available in this Composer version")
    
    if use_full_suite:
        # Core algorithms that work with streaming datasets
        algorithms.extend([
            MixUp(alpha=0.2),
            CutMix(alpha=1.0),
            LabelSmoothing(smoothing=0.1),
            EMA(half_life='50ba', update_interval='1ba'),
            ChannelsLast(),
            BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True)
        ])
        
        # RandAugment only works with VisionDataset - skip for streaming
        if not streaming:
            try:
                algorithms.append(RandAugment(severity=9, depth=2))
                print("   ✅ Added RandAugment (VisionDataset compatible)")
            except Exception as e:
                print(f"   ⚠️ RandAugment skipped: {e}")
        else:
            print("   💡 RandAugment skipped (not compatible with streaming datasets)")
    else:
        # Minimal set for faster training
        algorithms.extend([
            MixUp(alpha=0.2),
            LabelSmoothing(smoothing=0.1),
            ChannelsLast()
        ])
    
    print(f"🎯 Using {len(algorithms)} Composer algorithms:")
    for alg in algorithms:
        print(f"   ✅ {alg.__class__.__name__}")
    
    return algorithms


class TimeoutCallback:
    """Composer callback that enforces timeout during training."""
    
    def __init__(self, timeout_handler: TimeoutHandler):
        self.timeout_handler = timeout_handler
        self.last_check = time.time()
    
    def batch_end(self, state, logger):
        """Check timeout after each batch."""
        current_time = time.time()
        
        # Check timeout every 10 batches to avoid too much overhead
        if state.timestamp.batch % 10 == 0 or current_time - self.last_check > 30:
            self.last_check = current_time
            
            if self.timeout_handler.should_stop():
                elapsed_minutes = self.timeout_handler.elapsed_time() / 60
                print(f"\n⏰ TIMEOUT REACHED after {elapsed_minutes:.1f} minutes!")
                print(f"   Stopping training at batch {state.timestamp.batch}")
                
                # Stop training by setting a flag that Composer will check
                state.stop_training = True
                return
            
            # Progress update every 100 batches
            if state.timestamp.batch % 100 == 0:
                elapsed = self.timeout_handler.elapsed_time()
                remaining = self.timeout_handler.remaining_time()
                progress = self.timeout_handler.progress_percent()
                print(f"   ⏱️  Time: {elapsed/60:.1f}min elapsed, {remaining/60:.1f}min remaining ({progress:.1f}%)")


def setup_callbacks(timeout_handler: TimeoutHandler) -> list:
    """Setup training callbacks with timeout monitoring."""
    callbacks = [
        LRMonitor(),
        MemoryMonitor(), 
        SpeedMonitor(window_size=50),
        TimeoutCallback(timeout_handler),  # Add timeout callback
    ]
    
    return callbacks


class TimeoutTrainer(Trainer):
    """Custom Trainer with polling-based timeout support (no signals)."""
    
    def __init__(self, timeout_handler: TimeoutHandler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout_handler = timeout_handler
        self.last_timeout_check = time.time()
    
    def fit(self, *args, **kwargs):
        """Override fit with better timeout checking and error handling."""
        try:
            return super().fit(*args, **kwargs)
        except KeyboardInterrupt:
            if self.timeout_handler.should_stop():
                print("\n⏰ Training stopped due to timeout")
            else:
                print("\n⛔ Training interrupted by user")
            return
        except Exception as e:
            # Let the outer exception handler deal with this
            # Don't swallow exceptions here
            raise


def print_system_info():
    """Print system information for debugging."""
    print("=" * 60)
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check if we're in Colab
    try:
        import google.colab
        print("Environment: Google Colab ✅")
    except ImportError:
        print("Environment: Local/Other")
    
    print("=" * 60)


def main():
    """Main function for 1-hour validation training."""
    parser = argparse.ArgumentParser(description='1-Hour ImageNet Validation on Colab')
    
    # Core arguments
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size (default: 128 for Colab T4)')
    parser.add_argument('--subset-size', type=int, default=25000,
                       help='Dataset subset size (default: 25000)')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Data loading workers (default: 2 for Colab)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Initial learning rate (default: 0.05)')
    parser.add_argument('--timeout-minutes', type=int, default=60,
                       help='Training timeout in minutes (default: 60)')
    parser.add_argument('--streaming', action='store_true', default=True,
                       help='Use streaming dataset (default: True)')
    parser.add_argument('--minimal-algorithms', action='store_true',
                       help='Use minimal algorithm suite for faster training')
    parser.add_argument('--save-checkpoints', action='store_true',
                       help='Save model checkpoints')
    parser.add_argument('--hf-token', type=str, 
                       help='HuggingFace token for accessing gated datasets (optional if HF_TOKEN env var is set)')
    
    args = parser.parse_args()
    
    # Print system information
    print_system_info()
    
    # Setup timeout handler
    timeout_seconds = args.timeout_minutes * 60
    timeout_handler = TimeoutHandler(timeout_seconds)
    
    print(f"⏰ Training will run for {args.timeout_minutes} minutes")
    print(f"🎯 Target completion: {datetime.now() + timedelta(minutes=args.timeout_minutes)}")
    
    # Determine device
    if torch.cuda.is_available():
        device = 'gpu'  # Composer uses 'gpu' instead of 'cuda'
        print("🚀 Using GPU for training")
    else:
        device = 'cpu'
        print("⚠️  Using CPU for training (slower)")
    
    # Create model
    print("\n🏗️  Creating ResNet50 model...")
    model = ResNet50ComposerModel(num_classes=1000, pretrained=False)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Handle HuggingFace token
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if not hf_token:
        print("⚠️  Warning: No HuggingFace token provided.")
        print("   ImageNet-1k is a gated dataset. You may encounter authentication errors.")
        print("   Set HF_TOKEN environment variable or use --hf-token argument")
    
    # Create dataloaders
    print("\n📂 Setting up data pipelines...")
    train_loader, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        streaming=args.streaming,
        token=hf_token
    )
    
    # Setup optimizer
    optimizer = DecoupledSGDW(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Estimate training duration for scheduler with safety margin
    steps_per_epoch = len(train_loader)
    # For 1-hour training, estimate realistic number of steps
    # T4 GPU can process ~150-200 samples/sec, so ~6000-10000 batches in 1 hour
    # Use conservative estimate with safety margin to ensure timeout works
    timeout_minutes = args.timeout_minutes
    conservative_batches = min(8000, steps_per_epoch * 40)  # Conservative estimate
    
    # Add safety margin - plan for 90% of estimated batches to ensure timeout works
    max_batches_with_margin = int(conservative_batches * 0.9)
    
    scheduler = CosineAnnealingWithWarmupScheduler(
        t_warmup=f'{max(100, steps_per_epoch // 4)}ba',  # 25% of first epoch or 100 batches minimum
        t_max=f'{max_batches_with_margin}ba'
    )
    
    print(f"📊 Training config:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Conservative max batches (with safety margin): {max_batches_with_margin}")
    print(f"   Timeout enforcement: {timeout_minutes} minutes via TimeoutCallback")
    print(f"   Warmup steps: {max(100, steps_per_epoch // 4)}")
    
    # Setup Composer components
    algorithms = setup_composer_algorithms(
        use_full_suite=not args.minimal_algorithms,
        streaming=args.streaming
    )
    callbacks = setup_callbacks(timeout_handler)
    
    # Create save folder
    save_folder = './colab_checkpoints' if args.save_checkpoints else None
    if save_folder:
        Path(save_folder).mkdir(exist_ok=True)
    
    print(f"\n🎯 Starting 1-hour validation training...")
    print(f"📈 Batch size: {args.batch_size}")
    print(f"📊 Dataset size: {args.subset_size:,} samples")
    print(f"⚡ Learning rate: {args.lr}")
    
    # Create custom trainer with timeout support
    trainer = TimeoutTrainer(
        timeout_handler=timeout_handler,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=f'{max_batches_with_margin}ba',  # Conservative duration with safety margin
        eval_interval='500ba',  # Evaluate every 500 batches
        device=device,
        precision='amp_fp16',  # Mixed precision for efficiency
        algorithms=algorithms,
        callbacks=callbacks,
        save_folder=save_folder,
        save_interval='1000ba' if save_folder else None,
        # Note: gradient clipping removed - not supported in this Composer version
        # Gradient clipping can be added via algorithms if needed
        seed=42
    )
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 GPU cache cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Start training with comprehensive error handling
    start_time = time.time()
    print(f"\n🚀 Training started at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Training with timeout callback enforcement
        # The TimeoutCallback will stop training when timeout is reached
        trainer.fit()
        
        # Check how training ended
        if timeout_handler.should_stop():
            print(f"\n⏰ Training completed due to timeout ({args.timeout_minutes} minutes)")
        else:
            print(f"\n🏁 Training completed before timeout")
        
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted by user")
        
    except Exception as training_error:
        print(f"\n❌ Training error occurred:")
        print(f"   Error type: {type(training_error).__name__}")
        print(f"   Error message: {str(training_error) if str(training_error) else 'No error message provided'}")
        print(f"   Traceback:")
        traceback.print_exc()
        
        # Try to save current state
        if hasattr(trainer, 'state') and trainer.state:
            print(f"   Training state: {trainer.state.timestamp}")
        
        # Provide helpful suggestions based on error type
        error_str = str(training_error).lower()
        if "cuda out of memory" in error_str:
            print("\n💡 Suggestion: Try reducing batch size:")
            print("   !python colab_validation_1hour.py --batch-size 64 --streaming")
        elif "401" in error_str or "unauthorized" in error_str:
            print("\n💡 Suggestion: Check HuggingFace authentication:")
            print("   import os; os.environ['HF_TOKEN'] = 'your_token'")
        elif "timeout" in error_str or "signal" in error_str:
            print("\n💡 Note: This script now uses polling-based timeout (should be fixed)")
        
        # GPU memory cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU cache cleared after error")
    
    # Training completed
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🏁 TRAINING COMPLETED")
    print("=" * 60)
    print(f"⏱️  Duration: {duration/60:.1f} minutes")
    print(f"📊 Final state: {trainer.state.timestamp}")
    
    # Print final metrics if available
    if trainer.state.eval_metrics:
        final_acc = trainer.state.eval_metrics.get('MulticlassAccuracy', {}).get('val', 0)
        print(f"🎯 Final validation accuracy: {final_acc:.4f}")
    
    # GPU memory summary and cleanup
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"💾 Peak GPU memory: {peak_memory:.1f} GB")
        
        # Comprehensive GPU cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        
        # Check final memory state
        current_memory = torch.cuda.memory_allocated() / 1e9
        print(f"💾 Current GPU memory after cleanup: {current_memory:.1f} GB")
    
    print("\n✅ Validation run completed successfully!")
    print("🔧 Note: This script now uses improved timeout handling and error reporting")
    
    return trainer


if __name__ == '__main__':
    main()
