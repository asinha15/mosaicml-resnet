#!/usr/bin/env python3
"""
Test script to validate HuggingFace cache setup for ImageNet training.
Run this script to verify your local HF cache is working before training.
"""

import os
import sys
from pathlib import Path

# Add the shared directory to Python path
shared_dir = Path(__file__).parent / 'shared'
sys.path.insert(0, str(shared_dir))

def test_hf_cache_setup():
    """Test HuggingFace cache setup."""
    print("🔍 Testing HuggingFace Cache Setup")
    print("=" * 50)
    
    # Check environment variables
    print("1. Environment Variables:")
    hf_home = os.environ.get('HF_HOME')
    hf_hub_cache = os.environ.get('HF_HUB_CACHE')
    transformers_cache = os.environ.get('TRANSFORMERS_CACHE')
    offline_mode = os.environ.get('HF_DATASETS_OFFLINE')
    
    print(f"   HF_HOME: {hf_home}")
    print(f"   HF_HUB_CACHE: {hf_hub_cache}")
    print(f"   TRANSFORMERS_CACHE: {transformers_cache}")
    print(f"   HF_DATASETS_OFFLINE: {offline_mode}")
    
    if not hf_home:
        print("❌ HF_HOME not set")
        return False
    
    if offline_mode != '1':
        print("⚠️  HF_DATASETS_OFFLINE not set to '1' - will try to download from internet")
    
    # Check directory structure
    print("\n2. Directory Structure:")
    hf_home_path = Path(hf_home)
    
    if not hf_home_path.exists():
        print(f"❌ HF_HOME directory not found: {hf_home}")
        return False
    
    print(f"   ✅ HF_HOME exists: {hf_home}")
    
    # Check for ImageNet dataset
    imagenet_pattern = hf_home_path / "hub" / "datasets--imagenet-1k"
    imagenet_dirs = list(hf_home_path.glob("hub/datasets--imagenet-1k*"))
    
    if not imagenet_dirs:
        print("❌ ImageNet-1k dataset not found in cache")
        print(f"   Searched in: {imagenet_pattern}")
        return False
    
    print(f"   ✅ Found ImageNet cache: {imagenet_dirs[0]}")
    
    # Check cache size
    try:
        import subprocess
        result = subprocess.run(['du', '-sh', str(hf_home_path)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            cache_size = result.stdout.split()[0]
            print(f"   📊 Cache size: {cache_size}")
    except:
        print("   📊 Cache size: Unable to determine")
    
    # Test data loading
    print("\n3. Testing Data Loading:")
    try:
        from data_utils import create_dataloaders
        print("   ✅ Successfully imported data_utils")
        
        # Test with small subset
        train_loader, val_loader = create_dataloaders(
            batch_size=4,
            num_workers=0,  # Use single thread for testing
            subset_size=10,  # Very small subset
            use_hf=True,
            streaming=False  # Use cached data
        )
        
        print(f"   ✅ Created train loader: {len(train_loader)} batches")
        print(f"   ✅ Created val loader: {len(val_loader)} batches")
        
        # Test loading one batch
        print("   🔄 Loading test batch...")
        batch = next(iter(train_loader))
        images, labels = batch
        
        print(f"   ✅ Loaded batch: images {images.shape}, labels {labels.shape}")
        print(f"   ✅ Label range: {labels.min().item()}-{labels.max().item()}")
        print(f"   ✅ Image tensor type: {images.dtype}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_args():
    """Test that training arguments work correctly."""
    print("\n4. Testing Training Arguments:")
    
    try:
        from train import setup_args
        print("   ✅ Successfully imported train module")
        
        # Test arguments that should work with local cache
        test_args = [
            '--use-hf',
            '--data-subset', '100',
            '--batch-size', '8', 
            '--epochs', '1',
            '--dry-run'
        ]
        
        # Override sys.argv temporarily
        import sys
        original_argv = sys.argv
        sys.argv = ['train.py'] + test_args
        
        try:
            args = setup_args()
            print(f"   ✅ Parsed arguments successfully")
            print(f"   ✅ use_hf: {args.use_hf}")
            print(f"   ✅ streaming: {args.streaming}")
            print(f"   ✅ data_subset: {args.data_subset}")
            return True
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"   ❌ Training args test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 HuggingFace ImageNet Cache Validation")
    print("This script tests if your setup is ready for training")
    print()
    
    success = True
    
    # Run tests
    success &= test_hf_cache_setup()
    success &= test_training_args()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED!")
        print("🚀 Your setup is ready for training")
        print("\nTo start training, run:")
        print("   ./start_validation.sh")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please fix the issues above before training")
        
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
