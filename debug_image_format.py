#!/usr/bin/env python3
"""
Debug script to understand the exact format of image data in HuggingFace cache.
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

def analyze_image_format():
    """Analyze the exact format of image data."""
    print("🔍 Analyzing Image Data Format")
    print("=" * 40)
    
    try:
        from data_utils import find_cached_imagenet_data, load_cached_imagenet_dataset
        
        hf_home = os.environ['HF_HOME']
        cache_path = find_cached_imagenet_data(hf_home, 'train')
        
        if not cache_path:
            print("❌ No cached data found")
            return False
            
        # Load a small sample
        dataset = load_cached_imagenet_dataset(cache_path, 'train')
        
        if not dataset or len(dataset) == 0:
            print("❌ No dataset loaded")
            return False
            
        print(f"✅ Loaded dataset with {len(dataset)} samples")
        
        # Analyze first few samples
        for i in range(min(3, len(dataset))):
            print(f"\n📊 Sample {i}:")
            
            sample = dataset[i]
            print(f"   Sample keys: {list(sample.keys())}")
            
            if 'image' in sample:
                image_data = sample['image']
                print(f"   Image type: {type(image_data)}")
                
                if isinstance(image_data, dict):
                    print(f"   Image dict keys: {list(image_data.keys())}")
                    
                    for key, value in image_data.items():
                        print(f"   '{key}': {type(value)}, size: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                        
                        # If it's bytes, show first few bytes
                        if isinstance(value, bytes) and len(value) > 0:
                            print(f"      First 20 bytes: {value[:20]}")
                        elif isinstance(value, str):
                            print(f"      String value: '{value[:100]}{'...' if len(value) > 100 else ''}'")
                else:
                    print(f"   Image data: {str(image_data)[:100]}{'...' if len(str(image_data)) > 100 else ''}")
                
            if 'label' in sample:
                print(f"   Label: {sample['label']} (type: {type(sample['label'])})")
                
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_loading():
    """Test loading an image with the new format handler."""
    print(f"\n🔄 Testing Image Loading")
    print("-" * 40)
    
    try:
        from data_utils import ImageNetHF
        
        hf_home = os.environ['HF_HOME']
        
        # Create dataset instance
        dataset = ImageNetHF(
            split='train',
            subset_size=1,  # Just one sample
            transform=None,  # No transforms for testing
            streaming=False
        )
        
        if len(dataset) == 0:
            print("❌ Dataset is empty")
            return False
            
        print(f"✅ Created dataset with {len(dataset)} samples")
        
        # Try to load first sample
        print("🔄 Loading first sample...")
        image, label = dataset[0]
        
        print(f"✅ Successfully loaded sample:")
        print(f"   Image type: {type(image)}")
        print(f"   Image size: {image.size if hasattr(image, 'size') else 'Unknown'}")
        print(f"   Label: {label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 Image Format Debug Analysis")
    print("=" * 50)
    
    success1 = analyze_image_format()
    
    if success1:
        success2 = test_image_loading()
    else:
        success2 = False
    
    print(f"\n" + "=" * 50)
    if success1 and success2:
        print("✅ IMAGE FORMAT ANALYSIS COMPLETE!")
        print("🎉 Image loading should now work correctly")
    else:
        print("❌ IMAGE FORMAT ISSUES DETECTED")
        print("🔧 Need to adjust image handling code")
        
    print("=" * 50)
