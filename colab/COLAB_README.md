# 🚀 1-Hour ImageNet Validation Training for Google Colab

Complete validation training setup for testing ImageNet-1k training pipeline in exactly 1 hour on Google Colab.

## 🎯 Overview

This project provides a **time-controlled validation training** that:
- ✅ Streams ImageNet-1k data using HuggingFace datasets API
- ✅ Trains ResNet50 with MosaicML Composer optimizations
- ✅ Automatically terminates after exactly 60 minutes
- ✅ Is optimized for Google Colab T4 GPU (16GB VRAM)
- ✅ Validates your complete training pipeline end-to-end

## 📁 Files

- **`colab_validation_1hour.py`** - Complete Python script for command-line usage
- **`colab_validation_1hour.ipynb`** - Jupyter notebook for interactive Colab usage
- **`colab_requirements.txt`** - Dependencies for Colab environment
- **`COLAB_README.md`** - This guide

## 🚀 Quick Start (Recommended: Notebook)

### Prerequisites: HuggingFace Authentication 🔐

**⚠️ IMPORTANT**: ImageNet-1k is a gated dataset requiring authentication:

1. **Request Access**: Visit [imagenet-1k dataset](https://huggingface.co/datasets/imagenet-1k) and click "Request Access"
2. **Wait for Approval**: Usually takes 1-2 business days
3. **Get Token**: Go to [HuggingFace Settings](https://huggingface.co/settings/tokens) and create a new token
4. **Ready to Train!** ✅

### Option 1: Jupyter Notebook (Easiest)

1. **Upload to Colab**:
   ```bash
   # In Colab, upload colab_validation_1hour.ipynb
   ```

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator → T4 GPU

3. **Set your HF Token** in the authentication cell

4. **Run all cells** - the notebook handles everything automatically!

### Option 2: Python Script

1. **Setup Colab environment**:
   ```bash
   # Upload colab_validation_1hour.py to Colab
   
   # Method 1: Install from requirements (recommended)
   !pip install -r colab_requirements.txt
   
   # Method 2: If you encounter mosaicml metadata warnings, use:
   !pip install --upgrade pip
   !pip install "mosaicml>=0.20.0" --no-warn-conflicts
   !pip install datasets transformers huggingface_hub torchmetrics wandb
   ```

2. **Set Authentication**:
   ```python
   # Set your HuggingFace token
   import os
   os.environ['HF_TOKEN'] = 'your_token_here'  # Replace with your actual token
   ```

3. **Run training**:
   ```bash
   # Basic run (recommended settings)
   !python colab_validation_1hour.py
   
   # With token parameter
   !python colab_validation_1hour.py --hf-token your_token_here
   
   # Custom configuration
   !python colab_validation_1hour.py --batch-size 128 --subset-size 25000 --timeout-minutes 60
   ```

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 128 | Batch size (optimized for T4) |
| `--subset-size` | 25000 | ImageNet samples (2% of full dataset) |
| `--timeout-minutes` | 60 | Training duration in minutes |
| `--num-workers` | 2 | Data loading workers |
| `--lr` | 0.05 | Learning rate |
| `--streaming` | True | Use streaming dataset |
| `--minimal-algorithms` | False | Use fewer algorithms for speed |
| `--save-checkpoints` | False | Save model checkpoints |
| `--hf-token` | None | HuggingFace token (or set HF_TOKEN env var) |

## 📊 What to Expect

### Timeline (60 minutes total)
- **Setup & Installation**: 2-3 minutes
- **Data Loading**: 1-2 minutes  
- **Training**: 57-58 minutes
- **Results Analysis**: < 1 minute

### Performance Targets (T4 GPU)
- **GPU Utilization**: >85%
- **Memory Usage**: ~12-14 GB / 16 GB
- **Throughput**: ~150-200 samples/second
- **Batches Processed**: ~8,000-12,000 batches

### Expected Results
- **Training Loss**: Decreasing trend
- **Validation Accuracy**: 15-25% (limited by short training)
- **No Errors**: Complete pipeline validation
- **Stable Memory**: No memory leaks

## 🎯 Composer Optimizations Enabled

- ✅ **MixUp** - Data augmentation mixing
- ✅ **CutMix** - Patch-based augmentation  
- ✅ **RandAugment** - Automated augmentation
- ✅ **Label Smoothing** - Improved generalization
- ✅ **EMA** - Exponential moving averages
- ✅ **Channels Last** - Memory format optimization
- ✅ **BlurPool** - Anti-aliasing for better accuracy
- ✅ **Mixed Precision (FP16)** - 2x speed improvement

## 🔧 Troubleshooting

### Authentication Issues 🔐
```python
# Problem: 401 Unauthorized or access denied errors
# Solution 1: Check if you have access
# 1. Visit: https://huggingface.co/datasets/imagenet-1k
# 2. Make sure you see "✅ You have been granted access to this dataset"

# Solution 2: Set token correctly
import os
os.environ['HF_TOKEN'] = 'hf_your_actual_token_here'  # Replace with real token

# Solution 3: Interactive login (alternative)
from huggingface_hub import notebook_login
notebook_login()

# Test access:
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", split="train", streaming=True, token=os.environ.get('HF_TOKEN'))
```

### MosaicML Installation Issues 🔧
```python
# Problem: "WARNING: Ignoring version 0.12.0 of mosaicml since it has invalid metadata"
# This is a known issue with certain mosaicml versions

# Solution 1: Use latest version (recommended)
!pip install --upgrade pip
!pip install "mosaicml>=0.20.0" --no-warn-conflicts

# Solution 2: Install specific working version
!pip install "mosaicml==0.21.0" --force-reinstall

# Solution 3: If still failing, install from source
!pip install git+https://github.com/mosaicml/composer.git

# Verify installation works:
try:
    from composer import Trainer
    print("✅ MosaicML Composer installed successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
```

### Gradient Clipping Issues 🔧
```python
# Problem: "TypeError: Trainer.__init__() got an unexpected keyword argument 'grad_clip_norm'"
# The parameter name has changed in newer MosaicML versions

# The script now handles this automatically by:
# 1. Removing unsupported grad_clip_norm parameter
# 2. Adding GradientClipping algorithm if available
# 3. Falling back gracefully if not available

# Manual fix if needed:
# Remove grad_clip_norm from Trainer() parameters
```

### Scheduler Duration Mismatch 🔧
```python
# Problem: "ValueError: t_max 1950ba must be greater than or equal to max_duration 100ep"
# This happens when scheduler and trainer have mismatched duration units

# The script now handles this automatically by:
# 1. Estimating realistic batch count for 1-hour training
# 2. Setting both scheduler t_max and trainer max_duration to same value
# 3. Using batch units (ba) consistently for both

# Manual fix if needed:
# Ensure scheduler t_max >= trainer max_duration in same units
```

### RandAugment Dataset Compatibility 🔧
```python
# Problem: "To use RandAugment, the dataset must be a VisionDataset, not StreamingImageNetDataset"
# RandAugment algorithm requires VisionDataset inheritance

# The script now handles this automatically by:
# 1. Making StreamingImageNetDataset inherit from VisionDataset
# 2. Skipping RandAugment when using streaming datasets (safer approach)
# 3. Graceful fallback with other augmentation algorithms

# Algorithms used:
# - Streaming mode: MixUp, CutMix, LabelSmoothing, EMA, ChannelsLast, BlurPool
# - Non-streaming: Same + RandAugment
```

### Streaming Dataset Performance Issues 🔧
```python
# Problem: Training hangs at "0% 0/9750 [00:00<?, ?ba/s]" - no progress after several minutes
# This happens because streaming datasets have inefficient random access patterns

# The script now handles this automatically by:
# 1. Disabling shuffling for streaming datasets (shuffle=False)
# 2. Using single-threaded loading (num_workers=0)
# 3. Sequential data access with smart caching
# 4. VisionDataset inheritance for compatibility

# Manual fix if training hangs:
# 1. Stop training (Ctrl+C)
# 2. Run: !python restart_training_fixed.py

# Performance expectations:
# - Startup: 30-90 seconds (much faster than before)
# - First batch: Within 2 minutes
# - Steady throughput: 2-5 batches/second on T4 GPU
```

### Image Shape Mismatch Issues 🔧
```python
# Problem: "output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]"
# This happens when dataset contains grayscale images but model expects RGB

# The script now handles this automatically by:
# 1. Converting all images to RGB format: image.convert('RGB')
# 2. Ensuring consistent 3-channel input for the model
# 3. Proper handling of PIL Image modes

# Manual fix if needed:
# Add RGB conversion in your image processing:
if image.mode != 'RGB':
    image = image.convert('RGB')
```

### Memory Issues
```python
# Reduce batch size if OOM:
!python colab_validation_1hour.py --batch-size 64

# Use minimal algorithms:
!python colab_validation_1hour.py --minimal-algorithms
```

### GPU Not Detected
```python
# Check GPU availability:
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'None'}")

# Enable T4 GPU: Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

## 📈 Scaling to Full Training

After successful validation, scale up:

1. **Full Dataset**: Set `subset_size='full'` (1.28M samples)
2. **Longer Training**: Remove timeout, train for 90 epochs
3. **Better Hardware**: Use A100 or V100 GPUs
4. **Multi-GPU**: Implement distributed training
5. **Advanced Algorithms**: Add SAM, SWA for production

## 💡 Tips for Best Results

1. **GPU Runtime**: Always use T4 GPU in Colab
2. **Streaming**: Keep streaming=True for memory efficiency
3. **Batch Size**: 128 is optimal for T4, reduce if needed
4. **Monitoring**: Watch GPU utilization in Colab sidebar
5. **Checkpoints**: Enable saving for important runs
6. **Time Management**: Training auto-stops at 60 minutes

## 🎉 Success Criteria

✅ **Pipeline Validated** if you see:
- Training starts without errors
- GPU utilization >80%
- Loss decreases over time  
- No memory issues
- Automatic timeout after 1 hour
- Final metrics reported

## 📚 Next Steps

After validation success:
- Scale to full ImageNet training
- Experiment with hyperparameters
- Try different model architectures
- Implement production deployment

---

**Ready to validate your ImageNet training pipeline in 1 hour! 🚀**

*For questions or issues, check the troubleshooting section or refer to the MosaicML Composer documentation.*
