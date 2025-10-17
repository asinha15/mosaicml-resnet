#!/usr/bin/env python3
"""
Memory-safe cache loading test.
Tests ultra-conservative parquet loading approach.
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

def check_memory():
    """Check current memory status."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(':')] = int(parts[1])
        
        total_gb = meminfo['MemTotal'] / 1024 / 1024
        available_gb = meminfo['MemAvailable'] / 1024 / 1024
        used_gb = total_gb - available_gb
        
        print(f"💾 Memory Status:")
        print(f"   Total: {total_gb:.1f} GB")
        print(f"   Used: {used_gb:.1f} GB") 
        print(f"   Available: {available_gb:.1f} GB")
        
        return available_gb > 5.0  # Need at least 5GB free
        
    except Exception as e:
        print(f"❌ Could not check memory: {e}")
        return True  # Assume it's OK if we can't check

def test_single_file_loading():
    """Test loading a single parquet file to check memory usage."""
    print("🔍 Testing Single File Loading")
    print("-" * 40)
    
    try:
        from data_utils import find_cached_imagenet_data
        
        hf_home = os.environ['HF_HOME']
        cache_path = find_cached_imagenet_data(hf_home, 'train')
        
        if not cache_path:
            print("❌ No cached data found")
            return False
            
        print(f"📁 Cache path: {cache_path}")
        
        # Find parquet files
        cache_path_obj = Path(cache_path)
        parquet_files = list(cache_path_obj.glob("*.parquet"))
        
        if not parquet_files:
            print("❌ No parquet files found")
            return False
            
        print(f"📊 Found {len(parquet_files)} parquet files")
        
        # Test loading single file
        test_file = parquet_files[0]
        print(f"🔄 Testing file: {test_file.name}")
        
        import pandas as pd
        df = pd.read_parquet(test_file)
        
        print(f"✅ Loaded single file successfully:")
        print(f"   Samples: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check memory usage
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)  # MB
        print(f"   Memory usage: {memory_usage:.1f} MB")
        
        # Test sample access
        if len(df) > 0:
            sample = df.iloc[0]
            print(f"✅ Sample access successful:")
            if 'label' in sample:
                print(f"   Label: {sample['label']}")
            if 'image' in sample:
                print(f"   Image type: {type(sample['image'])}")
        
        # Clean up
        del df
        
        return True
        
    except Exception as e:
        print(f"❌ Single file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conservative_loading():
    """Test the new conservative loading approach."""
    print(f"\n🔄 Testing Conservative Data Loading")
    print("-" * 40)
    
    try:
        from data_utils import load_cached_imagenet_dataset, find_cached_imagenet_data
        
        hf_home = os.environ['HF_HOME'] 
        cache_path = find_cached_imagenet_data(hf_home, 'train')
        
        if not cache_path:
            print("❌ No cached data found")
            return False
            
        # Test the new conservative loading
        dataset = load_cached_imagenet_dataset(cache_path, 'train')
        
        if dataset:
            print(f"✅ Conservative loading successful:")
            print(f"   Dataset size: {len(dataset):,} samples")
            
            # Test sample access
            sample = dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            
            return True
        else:
            print("❌ Conservative loading returned None")
            return False
            
    except Exception as e:
        print(f"❌ Conservative loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 Memory-Safe Cache Loading Test")
    print("=" * 50)
    
    # Check memory first
    if not check_memory():
        print("⚠️  Low memory detected - proceed with caution")
    
    # Test single file loading
    test1 = test_single_file_loading()
    
    if test1:
        # Test conservative loading
        test2 = test_conservative_loading()
    else:
        test2 = False
    
    print(f"\n" + "=" * 50)
    if test1 and test2:
        print("✅ ALL MEMORY-SAFE TESTS PASSED!")
        print("🚀 Conservative loading should work for training")
    else:
        print("❌ MEMORY-SAFE TESTS FAILED")
        print("🔧 Need to investigate memory issues further")
        
    print("=" * 50)
