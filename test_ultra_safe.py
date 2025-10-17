#!/usr/bin/env python3
"""
Ultra-safe test with minimal memory usage.
Tests single file loading to avoid any memory issues.
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

def test_minimal_loading():
    """Test with absolutely minimal memory usage."""
    print("🔧 Ultra-Safe Minimal Loading Test")
    print("=" * 40)
    
    try:
        from data_utils import create_dataloaders
        
        print("🔄 Testing with minimal configuration...")
        print("   - Single worker")  
        print("   - Very small batch")
        print("   - Tiny subset")
        
        # Ultra-conservative settings
        train_loader, val_loader = create_dataloaders(
            batch_size=4,        # Very small batch
            num_workers=0,       # Single threaded
            subset_size=50,      # Very small subset
            use_hf=True,
            streaming=False,
            token=None
        )
        
        print(f"✅ Created dataloaders:")
        print(f"   Train: {len(train_loader)} batches")
        print(f"   Val: {len(val_loader)} batches")
        
        # Test loading a single batch
        print("🔄 Testing single batch loading...")
        batch = next(iter(train_loader))
        images, labels = batch
        
        print(f"✅ Batch loaded successfully:")
        print(f"   Images shape: {images.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Image dtype: {images.dtype}")
        print(f"   Labels range: {labels.min()}-{labels.max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 Ultra-Safe Memory Test")
    print("This test uses the most conservative settings possible")
    print("=" * 50)
    
    success = test_minimal_loading()
    
    print("=" * 50)
    if success:
        print("✅ ULTRA-SAFE TEST PASSED!")
        print("🎉 Basic data loading works - ready for larger tests")
    else:
        print("❌ ULTRA-SAFE TEST FAILED")
        print("🔧 Need to investigate fundamental data loading issues")
        
    print("=" * 50)
