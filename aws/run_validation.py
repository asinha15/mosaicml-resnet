#!/usr/bin/env python3
"""
AWS Validation Runner
Run this for a quick validation training on AWS to verify setup before full training.
"""
import sys
import os
from pathlib import Path

# Add shared directory to path
shared_dir = Path(__file__).parent.parent / "shared"
sys.path.insert(0, str(shared_dir))

from train import main
import argparse

def aws_validation_args():
    """Setup arguments for AWS validation run."""
    args = argparse.Namespace()
    
    # Model settings
    args.model_type = 'torchvision'
    args.pretrained = False
    args.compile_model = False
    
    # Data settings (AWS optimized)
    args.data_subset = '50000'  # Medium subset for validation
    args.batch_size = 256       # AWS GPU optimized
    args.image_size = 224
    args.num_workers = 8
    args.use_hf = True
    args.streaming = False      # Use non-streaming for deterministic validation
    
    # Training settings (2 epochs for validation)
    args.epochs = 2
    args.lr = 0.1
    args.weight_decay = 1e-4
    args.momentum = 0.9
    args.optimizer = 'sgd'
    
    # Composer algorithms (Full set)
    args.use_mixup = True
    args.use_cutmix = True
    args.use_randaugment = True
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
    
    # Logging
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
    print("🚀 Starting AWS Validation Run (2 epochs)")
    print("📊 Config: 50k samples, batch_size=256, streaming=False")
    print("🎯 This validates your AWS setup before full training")
    print()
    
    # Check for HF_TOKEN
    if not os.environ.get('HF_TOKEN'):
        print("⚠️  Warning: HF_TOKEN not set!")
        print("   Set it: export HF_TOKEN='your_token'")
        print("   Or you may encounter authentication errors with ImageNet-1k")
        print()
    
    args = aws_validation_args()
    
    try:
        main(args)
        print("✅ AWS validation completed successfully!")
        print("🎉 Your setup is ready for full training!")
        print()
        print("🚀 For full training, run:")
        print("   # Standard training:")
        print("   python ../shared/train.py --batch-size 256 --epochs 90 --data-subset full --wandb-project your-project")
        print()
        print("   # With streaming (faster startup):")
        print("   python ../shared/train.py --streaming --batch-size 256 --epochs 90 --data-subset full")
        
    except KeyboardInterrupt:
        print("\n⛔ AWS validation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ AWS validation failed:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e) if str(e) else 'No error message provided'}")
        
        # Provide helpful suggestions based on error type
        error_str = str(e).lower()
        if "cuda out of memory" in error_str:
            print("\n💡 Suggestions:")
            print("   - AWS instance may not have enough GPU memory")
            print("   - Try smaller batch size: modify run_validation.py")
            print("   - Ensure you're using a GPU instance (g4dn, p3, etc.)")
        elif "401" in error_str or "unauthorized" in error_str:
            print("\n💡 Suggestions:")
            print("   - Check HuggingFace authentication: export HF_TOKEN='your_token'")
            print("   - Verify ImageNet-1k dataset access approval")
        elif "dataset access error" in error_str or "cache" in error_str:
            print("\n💡 Suggestions:")
            print("   - Try validation without streaming")
            print("   - Check dataset configuration")
        elif "no module named" in error_str:
            print("\n💡 Suggestions:")
            print("   - Check AWS setup script completed successfully")
            print("   - Verify all dependencies are installed")
            print("   - Run: pip install -r ../shared/requirements.txt")
        
        print("\n🔍 For detailed debugging, check the full error above")
        sys.exit(1)
