#!/usr/bin/env python3
"""
🔧 Quick Setup Test for Colab ImageNet Validation

This script tests your environment before running the full 1-hour validation.
Run this first to ensure everything is properly configured.

Usage:
    python test_colab_setup.py
"""

import os
import sys
import time
from datetime import datetime

def test_python_version():
    """Test Python version compatibility."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires 3.8+)")
        return False

def test_pytorch():
    """Test PyTorch installation and CUDA."""
    print("\n🔥 Testing PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        # Test CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ CUDA: {torch.version.cuda}")
            print(f"   ✅ GPU: {gpu_name} ({memory_gb:.1f} GB)")
            
            # Test tensor operations
            x = torch.randn(100, 100).cuda()
            y = torch.mm(x, x.t())
            print(f"   ✅ GPU tensor operations working")
            return True, True
        else:
            print(f"   ⚠️ CUDA not available - will use CPU (very slow)")
            return True, False
            
    except ImportError as e:
        print(f"   ❌ PyTorch not installed: {e}")
        return False, False
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False, False

def test_composer():
    """Test MosaicML Composer installation."""
    print("\n🎵 Testing MosaicML Composer...")
    try:
        from composer import Trainer, ComposerModel
        from composer.algorithms import MixUp, LabelSmoothing
        print(f"   ✅ Composer imported successfully")
        
        # Test basic algorithm creation
        mixup = MixUp(alpha=0.2)
        label_smooth = LabelSmoothing(smoothing=0.1)
        print(f"   ✅ Composer algorithms working")
        return True
        
    except ImportError as e:
        print(f"   ❌ Composer not installed: {e}")
        print(f"   🔧 Try installing with: pip install 'mosaicml>=0.20.0' --no-warn-conflicts")
        print(f"   📖 See troubleshooting guide in COLAB_README.md")
        return False
    except Exception as e:
        print(f"   ❌ Composer error: {e}")
        if "metadata" in str(e).lower():
            print(f"   🔧 This looks like a version metadata issue.")
            print(f"   💡 Try: pip install 'mosaicml==0.21.0' --force-reinstall")
        return False

def test_datasets():
    """Test HuggingFace datasets and ImageNet access."""
    print("\n📊 Testing HuggingFace datasets...")
    try:
        from datasets import load_dataset
        print(f"   ✅ Datasets library imported")
        
        # Test ImageNet access (just metadata)
        print(f"   🔍 Testing ImageNet access...")
        
        # Check for HuggingFace token
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print(f"   ⚠️ No HF_TOKEN environment variable found")
            print(f"   💡 ImageNet-1k is a gated dataset requiring authentication")
            
        # Try to load dataset info without downloading
        try:
            dataset_info = load_dataset("imagenet-1k", split="train", streaming=True, token=hf_token)
            sample = next(iter(dataset_info.take(1)))
            print(f"   ✅ ImageNet access confirmed")
            print(f"   📷 Sample image: {sample['image'].size}")
            print(f"   🏷️ Sample label: {sample['label']}")
            if hf_token:
                print(f"   🔐 Authentication: Using HF_TOKEN")
            return True, True
            
        except Exception as e:
            print(f"   ❌ ImageNet access failed: {e}")
            if "401" in str(e) or "unauthorized" in str(e).lower():
                print(f"   🔐 AUTHENTICATION ERROR!")
                print(f"   Steps to fix:")
                print(f"      1. Visit: https://huggingface.co/datasets/imagenet-1k")
                print(f"      2. Request access (approval may take 1-2 days)")
                print(f"      3. Get token: https://huggingface.co/settings/tokens")
                print(f"      4. Set: export HF_TOKEN='your_token_here'")
            else:
                print(f"   💡 Check your internet connection and HuggingFace status")
            return True, False
            
    except ImportError as e:
        print(f"   ❌ Datasets not installed: {e}")
        return False, False
    except Exception as e:
        print(f"   ❌ Datasets error: {e}")
        return False, False

def test_torchvision():
    """Test torchvision and model loading."""
    print("\n👁️ Testing torchvision...")
    try:
        import torchvision.models as models
        import torchvision.transforms as transforms
        
        # Test ResNet50 loading
        model = models.resnet50(weights=None)
        print(f"   ✅ ResNet50 model loaded")
        
        # Test transforms
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ])
        print(f"   ✅ Transforms working")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   📊 ResNet50 parameters: {params:,}")
        return True
        
    except ImportError as e:
        print(f"   ❌ Torchvision not installed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Torchvision error: {e}")
        return False

def test_additional_packages():
    """Test additional required packages."""
    print("\n📦 Testing additional packages...")
    
    packages = [
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('tqdm', 'TQDM'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    all_good = True
    for pkg, name in packages:
        try:
            __import__(pkg)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} not installed")
            all_good = False
    
    return all_good

def test_memory_estimation():
    """Estimate memory requirements."""
    print("\n💾 Memory estimation...")
    try:
        import torch
        
        if torch.cuda.is_available():
            # Get GPU memory
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1e9
            
            print(f"   🖥️ GPU Memory: {total_memory:.1f} GB")
            
            # Estimate requirements
            model_memory = 0.1  # ~100MB for model
            batch_memory = 128 * 3 * 224 * 224 * 4 / 1e9  # Batch of 128 in FP32
            optimizer_memory = model_memory  # Optimizer states
            
            estimated_usage = model_memory + batch_memory + optimizer_memory
            print(f"   📊 Estimated usage: {estimated_usage:.1f} GB")
            
            if estimated_usage < total_memory * 0.9:  # 90% threshold
                print(f"   ✅ Memory sufficient for training")
                return True
            else:
                print(f"   ⚠️ Memory might be tight - consider smaller batch size")
                return True
        else:
            print(f"   ⚠️ No GPU detected - CPU training will be very slow")
            return True
            
    except Exception as e:
        print(f"   ❌ Memory check error: {e}")
        return False

def main():
    """Run all setup tests."""
    print("=" * 60)
    print("🔧 COLAB IMAGENET VALIDATION - SETUP TEST")
    print("=" * 60)
    print(f"🕒 Test started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Run all tests
    results = []
    
    results.append(("Python Version", test_python_version()))
    
    pytorch_ok, cuda_ok = test_pytorch()
    results.append(("PyTorch", pytorch_ok))
    results.append(("CUDA/GPU", cuda_ok))
    
    results.append(("MosaicML Composer", test_composer()))
    
    datasets_ok, imagenet_ok = test_datasets()
    results.append(("HF Datasets", datasets_ok))
    results.append(("ImageNet Access", imagenet_ok))
    
    results.append(("Torchvision", test_torchvision()))
    results.append(("Additional Packages", test_additional_packages()))
    results.append(("Memory Check", test_memory_estimation()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SETUP TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    # Overall result
    critical_tests = [r[1] for r in results[:6]]  # First 6 are critical
    all_critical_pass = all(critical_tests)
    
    print("\n" + "-" * 60)
    if all_critical_pass:
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("✅ Ready to run 1-hour ImageNet validation training")
        print("\n🚀 To start training, run:")
        print("   python colab_validation_1hour.py")
        print("   or use the Jupyter notebook: colab_validation_1hour.ipynb")
    else:
        print("❌ SOME CRITICAL TESTS FAILED!")
        print("⚠️ Fix the issues above before running validation training")
        print("\n💡 Common fixes:")
        print("   - Install missing packages: !pip install -r colab_requirements.txt")
        print("   - Enable GPU: Runtime → Change runtime type → Hardware accelerator → T4 GPU")
        print("   - Request ImageNet access: https://huggingface.co/datasets/imagenet-1k")
    
    print("\n" + "=" * 60)
    return all_critical_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
