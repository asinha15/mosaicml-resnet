#!/usr/bin/env python3
"""
Quick test script for cached data loading.
Tests the new memory-efficient parquet loading.
"""

import os
import sys
from pathlib import Path

# Set up environment
os.environ['HF_HOME'] = '/mnt/imagenet-data/hf_home'
os.environ['HF_HUB_CACHE'] = '/mnt/imagenet-data/hf_home/hub'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/imagenet-data/transformers_cache'

# Add shared directory to path
shared_dir = Path(__file__).parent / 'shared'
sys.path.insert(0, str(shared_dir))

def test_cache_loading():
    """Test cache loading with new memory-efficient approach."""
    print("🧪 Quick Cache Loading Test")
    print("=" * 40)
    
    try:
        from data_utils import find_cached_imagenet_data, load_cached_imagenet_dataset
        
        hf_home = os.environ['HF_HOME']
        print(f"📁 HF_HOME: {hf_home}")
        
        # Find cached data
        cache_path = find_cached_imagenet_data(hf_home, 'train')
        if not cache_path:
            print("❌ No cached data found")
            return False
            
        print(f"✅ Found cache path: {cache_path}")
        
        # Test loading with new approach
        print("\n🔄 Testing new memory-efficient loading...")
        dataset = load_cached_imagenet_dataset(cache_path, 'train')
        
        if dataset:
            print(f"✅ Successfully loaded dataset with {len(dataset)} samples")
            
            # Test accessing a sample
            sample = dataset[0]
            print(f"✅ Sample keys: {list(sample.keys())}")
            
            if 'image' in sample and 'label' in sample:
                print(f"✅ Sample contains required 'image' and 'label' fields")
                print(f"✅ Label: {sample['label']}")
                return True
            else:
                print(f"❌ Missing required fields in sample")
                return False
        else:
            print(f"❌ Failed to load dataset")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """Test creating a small dataloader."""
    print(f"\n🔄 Testing DataLoader creation...")
    
    try:
        from data_utils import create_dataloaders
        
        # Create small dataloaders for testing
        train_loader, val_loader = create_dataloaders(
            batch_size=8,
            num_workers=0,  # Single threaded for testing
            subset_size=100,  # Very small subset
            use_hf=True,
            streaming=False,
            token=None
        )
        
        print(f"✅ Created train loader: {len(train_loader)} batches")
        print(f"✅ Created val loader: {len(val_loader)} batches")
        
        # Test loading one batch
        batch = next(iter(train_loader))
        images, labels = batch
        print(f"✅ Loaded batch: images {images.shape}, labels {labels.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 Starting quick cache tests...")
    
    success1 = test_cache_loading()
    success2 = test_dataloader() if success1 else False
    
    print(f"\n" + "=" * 40)
    if success1 and success2:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Cache loading is working correctly")
        print("🚀 Ready to start training!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check the error messages above")
        
    print(f"=" * 40)
