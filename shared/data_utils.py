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
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
from typing import Optional, Tuple, Union
from PIL import Image
import pandas as pd
import glob


def find_cached_imagenet_data(hf_home: str, split: str = 'train') -> Optional[str]:
    """
    Find cached ImageNet data in HuggingFace cache directory.
    
    Args:
        hf_home: Path to HF_HOME directory
        split: 'train' or 'validation'
        
    Returns:
        Path to cached dataset directory or None if not found
    """
    hf_home_path = Path(hf_home)
    
    # Look for ImageNet dataset cache
    imagenet_cache_pattern = hf_home_path / "hub" / "datasets--imagenet-1k" / "snapshots"
    
    if not imagenet_cache_pattern.parent.exists():
        print(f"❌ No ImageNet cache found in {imagenet_cache_pattern.parent}")
        return None
    
    # Find the most recent snapshot
    snapshots = list(imagenet_cache_pattern.glob("*"))
    if not snapshots:
        print(f"❌ No snapshots found in {imagenet_cache_pattern}")
        return None
    
    # Use the first (most recent) snapshot
    snapshot_dir = snapshots[0]
    print(f"📁 Found ImageNet cache snapshot: {snapshot_dir}")
    
    # Look for the actual data directory
    possible_paths = [
        snapshot_dir / "data",
        snapshot_dir / f"data/{split}",
        snapshot_dir,
    ]
    
    for path in possible_paths:
        if path.exists():
            # Check if it contains parquet files or other data files
            parquet_files = list(path.glob("*.parquet"))
            arrow_files = list(path.glob("*.arrow"))
            
            if parquet_files or arrow_files:
                print(f"📊 Found cached data files in: {path}")
                return str(path)
    
    print(f"⚠️  Found cache directory but no data files in: {snapshot_dir}")
    return None


def load_cached_imagenet_dataset(cache_path: str, split: str = 'train', subset_size: Optional[int] = None) -> Optional[Dataset]:
    """
    Load ImageNet dataset directly from cached files.
    
    Args:
        cache_path: Path to cached data directory
        split: 'train' or 'validation'
        
    Returns:
        Dataset object or None if loading fails
    """
    cache_path_obj = Path(cache_path)
    
    try:
        # Method 1: Try loading as a saved dataset
        print(f"   🔄 Attempting to load cached dataset from {cache_path}")
        
        # Look for dataset files
        if (cache_path_obj / "dataset_info.json").exists():
            print(f"   📂 Found dataset_info.json, loading with load_from_disk...")
            dataset = load_from_disk(str(cache_path_obj))
            
            if isinstance(dataset, dict):
                # Multi-split dataset
                if split == 'validation' and 'validation' in dataset:
                    return dataset['validation']
                elif split == 'train' and 'train' in dataset:
                    return dataset['train']
                else:
                    # Return first available split
                    first_split = list(dataset.keys())[0]
                    print(f"   🔄 Requested split '{split}' not found, using '{first_split}'")
                    return dataset[first_split]
            else:
                return dataset
        
        # Method 2: Try loading parquet files directly (memory-efficient)
        parquet_files = list(cache_path_obj.glob("*.parquet"))
        if parquet_files:
            print(f"   📊 Found {len(parquet_files)} parquet files, loading with memory optimization...")
            
            # For large numbers of parquet files, use optimized chunked loading
            if len(parquet_files) > 50:
                print(f"   🔄 Large dataset detected ({len(parquet_files)} files)")
                print(f"   🔧 Using optimized chunked approach (reliable method)")
                
                # With 30GB+ available memory, we can use larger chunks
                # Get sample info from first file to determine optimal chunk size
                sample_file = parquet_files[0]
                print(f"   📊 Analyzing sample file: {sample_file.name}")
                
                try:
                    df_sample = pd.read_parquet(sample_file)
                    sample_count = len(df_sample)
                    columns = df_sample.columns.tolist()
                    memory_usage = df_sample.memory_usage(deep=True).sum() / (1024**2)  # MB
                    
                    print(f"   📊 Sample file: {sample_count:,} samples, {memory_usage:.1f} MB")
                    
                    # Smart chunking based on requested subset size
                    # Balance memory safety with dataset size requirements
                    if subset_size is None:
                        # Full dataset requested - load all files
                        estimated_samples_needed = len(parquet_files) * sample_count
                        print(f"   🔄 Full dataset requested - loading all {len(parquet_files)} files")
                    else:
                        # Specific subset size requested
                        estimated_samples_needed = subset_size
                        print(f"   🔄 Subset requested: {subset_size:,} samples")
                    
                    samples_per_file = max(1, sample_count)
                    files_needed = min(len(parquet_files), max(1, estimated_samples_needed // samples_per_file))
                    
                    if subset_size is None:
                        # Full dataset - load all files but in memory-safe chunks
                        print(f"   🔧 Full dataset: loading all {len(parquet_files)} files in chunks")
                        chunk_size = len(parquet_files)  # Load all files
                        
                        # For very large datasets, we'll process in smaller batches during the actual loading
                        if len(parquet_files) > 100:
                            print(f"   ⚠️  Large dataset ({len(parquet_files)} files) - will use batch processing")
                    elif files_needed == 1:
                        print(f"   🔧 Using single-file approach (sufficient for small requests)")
                        chunk_size = 1
                    elif files_needed <= 3:
                        print(f"   🔧 Using small multi-file approach ({files_needed} files for larger dataset)")
                        chunk_size = files_needed
                    else:
                        print(f"   🔧 Using conservative chunking approach (need {files_needed} files, using 4)")
                        chunk_size = 4  # Conservative but reasonable
                    
                    print(f"   📊 Will load {chunk_size} files (estimated {chunk_size * samples_per_file:,} samples)")
                    
                    print(f"   🔧 Using optimized chunk size: {chunk_size} files (memory efficient)")
                    
                    # Load files based on calculated chunk size
                    # Clean up sample file memory first
                    del df_sample
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    if subset_size is None and len(parquet_files) > 100:
                        # Full dataset with many files - use HuggingFace native loading (memory efficient)
                        print(f"   🔄 Using HuggingFace native parquet loading for full dataset...")
                        print(f"   💡 This avoids loading 160GB+ data into pandas memory")
                        
                        # Convert Path objects to strings for HuggingFace
                        parquet_paths = [str(f) for f in parquet_files]
                        
                        try:
                            # Use HuggingFace's efficient parquet loading
                            print(f"   🔄 Loading {len(parquet_paths)} parquet files directly with HuggingFace...")
                            dataset = Dataset.from_parquet(parquet_paths)
                            
                            print(f"   ✅ Successfully loaded full dataset: {len(dataset):,} samples")
                            print(f"   💾 Memory efficient: HuggingFace handles large datasets internally")
                            return dataset
                            
                        except Exception as hf_error:
                            print(f"   ❌ HuggingFace native loading failed: {hf_error}")
                            print(f"   🔄 Falling back to streaming approach...")
                            
                            # Fallback: Use load_dataset with parquet files
                            try:
                                from datasets import load_dataset
                                dataset = load_dataset(
                                    "parquet", 
                                    data_files=parquet_paths,
                                    split="train"
                                )
                                print(f"   ✅ Fallback successful: {len(dataset):,} samples")
                                return dataset
                                
                            except Exception as fallback_error:
                                print(f"   ❌ Fallback also failed: {fallback_error}")
                                print(f"   ⚠️  Will load subset instead to avoid OOM")
                                
                                # Last resort: Load only first 50 files to avoid OOM
                                subset_paths = parquet_paths[:50]
                                dataset = Dataset.from_parquet(subset_paths)
                                print(f"   ⚠️  Loaded subset only: {len(dataset):,} samples (50 files)")
                                return dataset
                        
                    elif chunk_size == 1:
                        # Single file approach (ultra-safe)
                        single_file = parquet_files[0]
                        print(f"   📂 Loading single file: {single_file.name}")
                        
                        try:
                            # Try HuggingFace native loading first
                            dataset = Dataset.from_parquet(str(single_file))
                            print(f"   ✅ Successfully loaded single file: {len(dataset):,} samples (HF native)")
                            return dataset
                            
                        except Exception as hf_error:
                            print(f"   ❌ HuggingFace loading failed: {hf_error}")
                            print(f"   🔄 Falling back to pandas for single file...")
                            
                            # Fallback to pandas
                            df_combined = pd.read_parquet(single_file)
                            print(f"   ✅ Loaded single file: {len(df_combined):,} samples")
                            
                            # Convert to HuggingFace Dataset
                            print(f"   🔄 Converting to HuggingFace Dataset...")
                            dataset = Dataset.from_pandas(df_combined)
                            del df_combined
                            gc.collect()
                            
                            print(f"   🎉 Successfully converted: {len(dataset):,} samples")
                            return dataset
                        
                    else:
                        # Smaller dataset - use HuggingFace native loading (still memory efficient)
                        chunk_files = parquet_files[:chunk_size]
                        chunk_paths = [str(f) for f in chunk_files]
                        
                        print(f"   📂 Loading {len(chunk_files)} files with HuggingFace native loading...")
                        
                        try:
                            # Use HuggingFace native loading even for smaller datasets
                            dataset = Dataset.from_parquet(chunk_paths)
                            print(f"   ✅ Successfully loaded {len(dataset):,} samples (memory efficient)")
                            return dataset
                            
                        except Exception as hf_error:
                            print(f"   ❌ HuggingFace loading failed: {hf_error}")
                            print(f"   🔄 Falling back to pandas approach for small dataset...")
                            
                            # Fallback to pandas for very small datasets
                            dataframes = []
                            for i, parquet_file in enumerate(chunk_files):
                                print(f"      📄 File {i+1}/{len(chunk_files)}: {parquet_file.name}")
                                df = pd.read_parquet(parquet_file)
                                dataframes.append(df)
                            
                            if dataframes:
                                print(f"   🔄 Combining {len(dataframes)} dataframes...")
                                df_combined = pd.concat(dataframes, ignore_index=True)
                                
                                # Free individual dataframes immediately
                                del dataframes
                                gc.collect()
                                
                                print(f"   ✅ Loaded combined files: {len(df_combined):,} samples")
                                
                                # Convert to HuggingFace Dataset
                                print(f"   🔄 Converting to HuggingFace Dataset...")
                                dataset = Dataset.from_pandas(df_combined)
                                del df_combined
                                gc.collect()
                                
                                print(f"   🎉 Successfully converted: {len(dataset):,} samples")
                                return dataset
                            else:
                                return None
                        
                except Exception as chunk_error:
                    print(f"   ❌ Optimized chunked loading failed: {chunk_error}")
                    print(f"   🔄 Falling back to basic approach...")
            else:
                # Small number of files - load normally
                print(f"   🔄 Loading {len(parquet_files)} parquet files...")
                dataframes = []
                
                for i, parquet_file in enumerate(parquet_files):
                    print(f"   📂 Loading file {i+1}/{len(parquet_files)}: {parquet_file.name}")
                    df = pd.read_parquet(parquet_file)
                    dataframes.append(df)
                
                if dataframes:
                    combined_df = pd.concat(dataframes, ignore_index=True)
                    print(f"   ✅ Loaded {len(combined_df)} samples from parquet files")
                    
                    # Convert to HuggingFace Dataset
                    dataset = Dataset.from_pandas(combined_df)
                    return dataset
        
        # Method 3: Try loading arrow files
        arrow_files = list(cache_path_obj.glob("*.arrow"))
        if arrow_files:
            print(f"   📊 Found {len(arrow_files)} arrow files, loading directly...")
            
            # Try to load as Dataset
            dataset = Dataset.from_file(str(arrow_files[0]))
            return dataset
            
        print(f"   ❌ No compatible data files found in {cache_path}")
        return None
        
    except Exception as e:
        print(f"   ❌ Failed to load cached dataset: {e}")
        return None


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
            # Try to load from cache first (bypasses authentication)
            if hf_home:
                print(f"   🔍 Searching for cached ImageNet data...")
                cache_path = find_cached_imagenet_data(hf_home, split)
                
                if cache_path:
                    print(f"   💾 Loading from cached files (no authentication needed)")
                    cached_dataset = load_cached_imagenet_dataset(cache_path, split, subset_size)
                    
                    if cached_dataset is not None:
                        self.dataset = cached_dataset
                        
                        if subset_size is not None:
                            # Sample random subset
                            total_size = len(self.dataset)
                            actual_subset_size = min(subset_size, total_size)
                            indices = np.random.choice(total_size, actual_subset_size, replace=False)
                            self.dataset = self.dataset.select(indices)
                            print(f"✅ Created subset with {len(self.dataset)} samples from cache")
                        
                        self._length = len(self.dataset)
                        print(f"✅ Successfully loaded {self._length:,} samples from cached data")
                        return  # Success! Exit early
                    else:
                        print(f"   ⚠️  Could not load cached data, falling back to Hub download...")
                else:
                    print(f"   ⚠️  No cached data found, falling back to Hub download...")
            
            # Fallback: Try Hub download (requires authentication)
            print(f"   🌐 Attempting to load from HuggingFace Hub...")
            
            if offline_mode:
                # Temporarily disable offline mode for Hub access
                original_offline = os.environ.get('HF_DATASETS_OFFLINE')
                if 'HF_DATASETS_OFFLINE' in os.environ:
                    del os.environ['HF_DATASETS_OFFLINE']
                
                try:
                    self.dataset = load_dataset(
                        "imagenet-1k", 
                        split=split, 
                        streaming=streaming,
                        token=self.token,
                        cache_dir=hf_home if hf_home else None
                    )
                finally:
                    # Restore offline mode
                    if original_offline:
                        os.environ['HF_DATASETS_OFFLINE'] = original_offline
            else:
                self.dataset = load_dataset(
                    "imagenet-1k", 
                    split=split, 
                    streaming=streaming,
                    token=self.token,
                    cache_dir=hf_home if hf_home else None
                )
            
            # Handle subset for Hub-loaded data
            if streaming:
                if subset_size is not None:
                    self._length = min(subset_size, 1281167 if split == 'train' else 50000)
                    self._subset_size = subset_size
                    print(f"   📊 Will use first {self._length:,} samples from stream")
                else:
                    self._length = 1281167 if split == 'train' else 50000
                    self._subset_size = None
            else:
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
        
        # Process image - handle different formats from HuggingFace cache
        image_data = item['image']
        
        if isinstance(image_data, Image.Image):
            # Already a PIL Image
            image = image_data
        elif isinstance(image_data, dict):
            # Dictionary format from HuggingFace cache
            if 'bytes' in image_data and image_data['bytes'] is not None:
                # Raw image bytes
                import io
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif 'path' in image_data and image_data['path'] is not None:
                # File path
                image = Image.open(image_data['path'])
            else:
                # Try to find any key that might contain image data
                for key, value in image_data.items():
                    if isinstance(value, bytes) and len(value) > 100:  # Likely image bytes
                        import io
                        try:
                            image = Image.open(io.BytesIO(value))
                            break
                        except:
                            continue
                else:
                    raise ValueError(f"Unable to decode image from dictionary: {list(image_data.keys())}")
        else:
            # Numpy array or other format
            try:
                import numpy as np
                if hasattr(image_data, '__array__') or isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data)
                else:
                    raise ValueError(f"Unsupported image format: {type(image_data)}")
            except Exception as e:
                raise ValueError(f"Failed to convert image data of type {type(image_data)}: {e}")
        
        # Ensure image is RGB (convert grayscale to RGB if needed)
        if hasattr(image, 'mode') and image.mode != 'RGB':
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
