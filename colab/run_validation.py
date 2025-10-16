#!/usr/bin/env python3
"""
Colab Validation Runner
Run this for a quick 1-hour validation training on Colab to verify setup.
"""
import sys
import os
from pathlib import Path

# Add shared directory to path
shared_dir = Path(__file__).parent.parent / "shared"
sys.path.insert(0, str(shared_dir))

from train import main
import argparse

def colab_validation_args():
    """Setup arguments for Colab validation run."""
    args = argparse.Namespace()
    
    # Model settings
    args.model_type = 'torchvision'
    args.pretrained = False
    args.compile_model = False
    
    # Data settings (optimized for Colab)
    args.data_subset = '25000'  # Small subset for validation
    args.batch_size = 128       # Colab T4 optimized
    args.image_size = 224
    args.num_workers = 2
    args.use_hf = True
    args.streaming = True       # Use streaming for faster startup
    
    # Training settings (1 hour validation)
    args.epochs = 1
    args.lr = 0.05
    args.weight_decay = 1e-4
    args.momentum = 0.9
    args.optimizer = 'sgd'
    
    # Composer algorithms (Colab optimized)
    args.use_mixup = True
    args.use_cutmix = True
    args.use_randaugment = False  # Skip RandAugment for streaming
    args.use_label_smoothing = True
    args.use_ema = True
    args.use_channels_last = True
    args.use_blurpool = True
    args.use_sam = False
    args.use_swa = False
    
    # Infrastructure
    args.device = 'auto'
    args.precision = 'amp_fp16'
    args.save_folder = './checkpoints'
    args.save_interval = '1ep'
    
    # Logging (minimal for Colab)
    args.log_level = 'INFO'
    args.wandb_project = None
    args.wandb_group = None
    args.wandb_name = None
    args.wandb_tags = []
    
    # Other
    args.seed = 42
    args.dry_run = False
    args.find_lr = False
    
    return args

if __name__ == "__main__":
    print("🚀 Starting Colab Validation Run (1 hour)")
    print("📊 Config: 25k samples, batch_size=128, streaming=True")
    print("🎯 This validates your Colab setup before full training")
    print()
    
    # Check for HF_TOKEN
    if not os.environ.get('HF_TOKEN'):
        print("⚠️  Warning: HF_TOKEN not set!")
        print("   Set it in Colab: import os; os.environ['HF_TOKEN'] = 'your_token'")
        print("   Or you may encounter authentication errors with ImageNet-1k")
        print()
    
    args = colab_validation_args()
    
    try:
        main(args)
        print("✅ Colab validation completed successfully!")
        print("🎉 Your setup is ready for full training!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        print("💡 Check the error above and fix any setup issues")
        sys.exit(1)
