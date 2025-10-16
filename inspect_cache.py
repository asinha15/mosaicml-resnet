#!/usr/bin/env python3
"""
Simple cache inspection script to understand HuggingFace ImageNet cache structure.
This helps debug what's actually in your cache directory.
"""

import os
from pathlib import Path
import sys

def inspect_cache_structure():
    """Inspect the HuggingFace cache structure in detail."""
    
    print("🔍 HuggingFace Cache Inspector")
    print("=" * 50)
    
    # Get cache directory
    hf_home = os.environ.get('HF_HOME', '/mnt/imagenet-data/hf_home')
    print(f"📁 HF_HOME: {hf_home}")
    
    if not Path(hf_home).exists():
        print(f"❌ HF_HOME directory not found: {hf_home}")
        return False
    
    hf_home_path = Path(hf_home)
    
    # Look for ImageNet cache
    print(f"\n🔍 Searching for ImageNet cache...")
    imagenet_cache = hf_home_path / "hub" / "datasets--imagenet-1k"
    
    if not imagenet_cache.exists():
        print(f"❌ ImageNet cache not found at: {imagenet_cache}")
        
        # List what's actually in the hub directory
        hub_dir = hf_home_path / "hub"
        if hub_dir.exists():
            print(f"\n📂 Contents of hub directory:")
            for item in hub_dir.iterdir():
                print(f"   📄 {item.name}")
        
        return False
    
    print(f"✅ Found ImageNet cache: {imagenet_cache}")
    
    # Examine snapshots
    snapshots_dir = imagenet_cache / "snapshots"
    if not snapshots_dir.exists():
        print(f"❌ No snapshots directory found")
        return False
    
    snapshots = list(snapshots_dir.glob("*"))
    print(f"\n📂 Found {len(snapshots)} snapshot(s):")
    
    for i, snapshot in enumerate(snapshots):
        print(f"\n📄 Snapshot {i+1}: {snapshot.name}")
        
        # List contents of snapshot
        items = list(snapshot.iterdir())
        print(f"   📊 Contains {len(items)} items:")
        
        for item in items[:10]:  # Show first 10 items
            if item.is_file():
                size = item.stat().st_size / (1024*1024)  # MB
                print(f"      📄 {item.name} ({size:.1f} MB)")
            else:
                print(f"      📁 {item.name}/")
                
                # If it's a directory, show its contents too
                if item.name == "data":
                    sub_items = list(item.iterdir())
                    print(f"         📊 Data directory contains {len(sub_items)} items:")
                    
                    # Show file types and sizes
                    file_types = {}
                    total_size = 0
                    
                    for sub_item in sub_items:
                        if sub_item.is_file():
                            ext = sub_item.suffix.lower() or 'no_extension'
                            size = sub_item.stat().st_size / (1024*1024)  # MB
                            
                            if ext not in file_types:
                                file_types[ext] = {'count': 0, 'total_size': 0}
                            
                            file_types[ext]['count'] += 1
                            file_types[ext]['total_size'] += size
                            total_size += size
                    
                    for ext, info in file_types.items():
                        avg_size = info['total_size'] / info['count'] if info['count'] > 0 else 0
                        print(f"         📄 {ext}: {info['count']} files, "
                              f"{info['total_size']:.1f} MB total, "
                              f"{avg_size:.1f} MB avg")
                    
                    print(f"         💾 Total data size: {total_size:.1f} MB")
                    
                    # Show a few sample files
                    sample_files = sub_items[:5]
                    print(f"         📄 Sample files:")
                    for sample in sample_files:
                        if sample.is_file():
                            print(f"            📄 {sample.name}")
        
        if len(items) > 10:
            print(f"      ... and {len(items) - 10} more items")
    
    print(f"\n✅ Cache inspection completed!")
    
    # Try to identify data format
    print(f"\n🔍 Data Format Analysis:")
    
    # Look for specific file types that indicate the data format
    for snapshot in snapshots:
        data_dir = snapshot / "data"
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            arrow_files = list(data_dir.glob("*.arrow"))
            json_files = list(data_dir.glob("*.json"))
            
            if parquet_files:
                print(f"   📊 Found parquet files: {len(parquet_files)} files")
                print(f"      💡 This suggests cached dataset in parquet format")
                
                # Try to read a parquet file
                try:
                    import pandas as pd
                    sample_parquet = parquet_files[0]
                    df = pd.read_parquet(sample_parquet)
                    print(f"      ✅ Sample parquet file has {len(df)} rows and columns: {list(df.columns)}")
                    
                    # Show first few rows info
                    if 'image' in df.columns:
                        print(f"      📷 Image column type: {type(df['image'].iloc[0])}")
                    if 'label' in df.columns:
                        print(f"      🏷️  Label column sample: {df['label'].head().tolist()}")
                        
                except Exception as e:
                    print(f"      ❌ Could not read parquet file: {e}")
            
            if arrow_files:
                print(f"   🏹 Found arrow files: {len(arrow_files)} files")
                print(f"      💡 This suggests cached dataset in arrow format")
            
            if json_files:
                print(f"   📄 Found JSON files: {len(json_files)} files")
                for json_file in json_files:
                    print(f"      📄 {json_file.name}")
    
    return True

if __name__ == '__main__':
    inspect_cache_structure()
