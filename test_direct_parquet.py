#!/usr/bin/env python3
"""
Direct parquet loading test without HuggingFace Dataset conversion.
This bypasses the problematic conversion step to isolate the image format issues.
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

def test_direct_parquet():
    """Test loading parquet directly without HuggingFace conversion."""
    print("🔄 Direct Parquet Loading Test")
    print("=" * 40)
    
    try:
        from data_utils import find_cached_imagenet_data
        import pandas as pd
        from PIL import Image
        import io
        
        hf_home = os.environ['HF_HOME']
        cache_path = find_cached_imagenet_data(hf_home, 'train')
        
        if not cache_path:
            print("❌ No cached data found")
            return False
            
        # Find parquet files
        cache_path_obj = Path(cache_path)
        parquet_files = list(cache_path_obj.glob("*.parquet"))
        
        if not parquet_files:
            print("❌ No parquet files found")
            return False
            
        print(f"✅ Found {len(parquet_files)} parquet files")
        
        # Load just one file
        test_file = parquet_files[0]
        print(f"🔄 Loading: {test_file.name}")
        
        df = pd.read_parquet(test_file)
        print(f"✅ Loaded DataFrame: {len(df)} samples")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Test accessing image data directly
        for i in range(min(3, len(df))):
            print(f"\n📊 Testing sample {i}:")
            
            sample = df.iloc[i]
            image_data = sample['image']
            label = sample['label']
            
            print(f"   Label: {label}")
            print(f"   Image type: {type(image_data)}")
            
            if isinstance(image_data, dict):
                print(f"   Image dict keys: {list(image_data.keys())}")
                
                # Try to decode the image
                try:
                    if 'bytes' in image_data and image_data['bytes']:
                        image_bytes = image_data['bytes']
                        print(f"   Image bytes length: {len(image_bytes)}")
                        
                        # Try to open as PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        print(f"   ✅ Successfully decoded image: {image.size}, {image.mode}")
                        
                    elif 'path' in image_data and image_data['path']:
                        image_path = image_data['path']
                        print(f"   Image path: {image_path}")
                        
                        image = Image.open(image_path)
                        print(f"   ✅ Successfully loaded from path: {image.size}, {image.mode}")
                        
                    else:
                        # Look for any bytes-like data
                        for key, value in image_data.items():
                            if isinstance(value, bytes) and len(value) > 100:
                                try:
                                    image = Image.open(io.BytesIO(value))
                                    print(f"   ✅ Decoded from '{key}': {image.size}, {image.mode}")
                                    break
                                except Exception as e:
                                    print(f"   ❌ Failed to decode '{key}': {e}")
                        else:
                            print(f"   ❌ No decodable image data found")
                            
                except Exception as decode_error:
                    print(f"   ❌ Image decode failed: {decode_error}")
            else:
                print(f"   Image data preview: {str(image_data)[:100]}")
                
        return True
        
    except Exception as e:
        print(f"❌ Direct parquet test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 Direct Parquet Access Test")
    print("This test bypasses HuggingFace Dataset conversion")
    print("=" * 50)
    
    success = test_direct_parquet()
    
    print("=" * 50)
    if success:
        print("✅ DIRECT PARQUET TEST PASSED!")
        print("🎉 Image data can be decoded - format is understood")
    else:
        print("❌ DIRECT PARQUET TEST FAILED")
        print("🔧 Image format issues need investigation")
        
    print("=" * 50)
