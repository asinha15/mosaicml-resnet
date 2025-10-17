# 🚀 Production-Ready ResNet50 ImageNet Training

**High-performance ResNet50 training on ImageNet using MosaicML Composer with GPU optimization, memory-safe data loading, and multi-platform support.**

## ⚡ **Quick Start**

### **🎯 Option 1: AWS Training (Recommended for Production)**
```bash
# 1. Launch AWS instance with ImageNet data
./aws/scripts/user-data-training.sh

# 2. Set your API keys
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"  # Optional

# 3. Choose your configuration:
./start_validation.sh                    # 1-hour validation (20K samples)
./start_training.sh aws_a10g_max_config  # Maximum GPU utilization (50K samples)
```

### **🆓 Option 2: Google Colab (Free GPU)**
```python
# In Colab notebook:
!git clone https://github.com/your-username/mosaic-resnet.git
%cd mosaic-resnet
!pip install -r shared/requirements.txt

# Set your token and run
import os
os.environ['HF_TOKEN'] = 'your_token'
!python shared/train.py --use-hf --streaming --data-subset 5000 --epochs 3 --batch-size 64
```

## 🎛️ **Training Configurations**

| Configuration | Dataset Size | Batch Size | Duration | GPU Memory | Best For |
|---------------|--------------|------------|----------|------------|----------|
| `aws_g4dn_validation_config` | 20K samples | 256 | ~1 hour | ~12GB | **Validation & testing** |
| `aws_a10g_max_config` | 50K samples | 384 | ~1 hour | ~18GB | **Maximum A10G utilization** |
| `aws_g4dn_12xl_ddp_config` | Full ImageNet | 128×4 | ~20 hours | ~14GB×4 | **Production training** |
| Custom Colab | 1K-10K samples | 64-128 | 10-60 min | ~8-15GB | **Experimentation** |

## 🛠️ **Usage Examples**

### **🚀 AWS High-Performance Training**
```bash
# Optimized 1-hour validation (recommended first run)
./start_validation.sh

# Maximum A10G GPU utilization  
./start_training.sh aws_a10g_max_config

# Check GPU optimization opportunities
./restart_optimized.sh

# Custom configuration
./start_training.sh aws_g4dn_validation_config
```

### **💻 Local/Development Training**
```bash
cd shared

# Quick test (5 minutes)
python train.py --use-hf --streaming --data-subset 1000 --epochs 1 --batch-size 64

# Medium validation (30 minutes)  
python train.py --use-hf --data-subset 10000 --epochs 5 --batch-size 128

# Custom configuration
python train.py --use-hf --data-subset 25000 --epochs 10 --batch-size 256 \
  --lr 0.1 --use-mixup --use-cutmix --use-ema --compile-model
```

### **🔧 GPU Optimization Helper**
```bash
# Analyze current GPU utilization and get recommendations
./restart_optimized.sh

# Example output:
# 📊 Current GPU Status: 4.4GB/23GB (19% memory), 52% utilization
# 🚀 MASSIVE HEADROOM DETECTED!
# → Recommended: aws_a10g_max_config (384 batch size, 50K samples)
```

## 🏗️ **Project Structure**

```
mosaic-resnet/
├── 🚀 aws/                          # AWS deployment & scripts
│   ├── scripts/
│   │   ├── user-data-training.sh       # Complete AWS setup & training
│   │   └── user-data-download.sh       # ImageNet data download
│   ├── DEPLOYMENT.md                   # AWS deployment guide
│   └── HUGGINGFACE_SETUP.md           # Authentication setup
├── 💻 colab/                        # Google Colab support  
│   ├── colab_validation_1hour.ipynb    # Jupyter notebook
│   └── colab_requirements.txt          # Colab dependencies
├── 🧠 shared/                       # Core training code
│   ├── train.py                       # Main training script ⭐
│   ├── data_utils.py                  # Memory-optimized data loading ⭐
│   ├── model.py                       # ResNet50 model definitions
│   ├── evaluate.py                    # Model evaluation utilities
│   └── configs/training_configs.yaml  # Training configurations
├── 📊 restart_optimized.sh          # GPU optimization helper ⭐
├── 📋 PROJECT_SUMMARY.md            # Technical project overview
└── 📖 README.md                     # This file
```

## ⚙️ **Command Line Options**

### **Core Arguments**
```bash
--use-hf                 # Use HuggingFace ImageNet dataset (recommended)
--streaming             # Stream data instead of downloading (faster startup)
--data-subset N         # Number of samples (or 'full' for complete ImageNet)
--batch-size N          # Batch size (adjust based on GPU memory)
--epochs N              # Number of training epochs
--lr FLOAT              # Learning rate
--compile-model         # Enable torch.compile() for 15-30% speedup
```

### **Optimization Flags**
```bash
--use-mixup             # MixUp data augmentation
--use-cutmix            # CutMix data augmentation  
--use-randaugment       # RandAugment (disabled for streaming)
--use-label-smoothing   # Label smoothing
--use-ema               # Exponential Moving Average
--use-channels-last     # Memory-efficient format
--use-blurpool          # Anti-aliasing for better accuracy
--precision amp_fp16    # Mixed precision training
```

### **Logging & Monitoring**
```bash
--wandb-project NAME    # Weights & Biases project name
--log-interval N        # Logging frequency (batches)
--save-interval N       # Checkpoint saving frequency
--save-folder PATH      # Checkpoint directory
```

## 🚀 **Performance Optimizations**

### **🎯 Memory-Safe Data Loading**
- **Smart chunking**: Automatically adapts to available memory
- **HF cache support**: Direct loading from local ImageNet cache  
- **Fallback strategies**: Multiple memory optimization levels
- **Garbage collection**: Aggressive cleanup prevents OOM errors

### **⚡ GPU Optimizations**
- **torch.compile()**: 15-30% speed improvement
- **Mixed precision**: AMP FP16 for memory efficiency
- **Channels-last**: Optimized memory layout
- **Dynamic batching**: Automatic batch size optimization

### **📊 Training Features**
- **Multiple algorithms**: MixUp, CutMix, EMA, Label Smoothing
- **Learning rate scheduling**: Cosine annealing with warmup
- **Checkpointing**: Automatic model saving and resuming
- **Comprehensive logging**: WandB integration with system monitoring

## 🔧 **Troubleshooting**

### **🚨 Common Issues & Solutions**

#### **Out of Memory (OOM)**
```bash
# Symptoms: CUDA out of memory, process killed
# Solutions:
./restart_optimized.sh                    # Check GPU utilization
python train.py --batch-size 64           # Reduce batch size
python train.py --num-workers 2           # Reduce data workers
python train.py --data-subset 5000        # Use smaller dataset
```

#### **Slow Training / Low GPU Utilization**
```bash
# Check current utilization
./restart_optimized.sh

# If <70% GPU utilization, optimize:
./start_training.sh aws_a10g_max_config   # Use optimized config
python train.py --compile-model            # Enable torch.compile
python train.py --batch-size 256          # Increase batch size
python train.py --num-workers 8           # More data workers
```

#### **Authentication Errors**
```bash
# Get HuggingFace token: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc

# Request ImageNet access: https://huggingface.co/datasets/imagenet-1k
# (Usually approved within 24-48 hours)
```

#### **Data Loading Issues**
```bash
# Use streaming for faster startup
python train.py --use-hf --streaming

# Or use local cache (AWS instances with mounted data)  
python train.py --use-hf  # Automatically detects local cache
```

## 📊 **Expected Performance**

### **Validation Runs (1 hour)**
- **Loss**: Decreases from ~6.9 to ~4.5
- **Accuracy**: Improves from ~0.1% to ~15-25%
- **GPU Utilization**: 75-95% (optimized configs)
- **Memory Usage**: 10-20GB depending on batch size

### **Full Training (90 epochs)**
- **Top-1 Accuracy**: 76-78% on ImageNet validation
- **Top-5 Accuracy**: 93-95% on ImageNet validation
- **Training Time**: 15-25 hours (4x T4 GPUs)
- **Final Loss**: 0.8-1.2 cross-entropy

### **Hardware Performance**
| GPU | Batch Size | Speed | Memory | Training Time (90 epochs) |
|-----|------------|-------|--------|---------------------------|
| T4 (Colab) | 64-128 | 2-4 ba/s | 8-15GB | ~80-120 hours |
| T4 (AWS g4dn.xlarge) | 128-256 | 3-6 ba/s | 12-16GB | ~60-80 hours |
| A10G (AWS g5.xlarge) | 256-384 | 8-12 ba/s | 15-20GB | ~25-35 hours |
| 4× T4 (AWS g4dn.12xlarge) | 512-1024 | 15-25 ba/s | 50-60GB | ~15-25 hours |

## 🎯 **Quick Validation Checklist**

### **✅ First-Time Setup**
1. **Get HuggingFace token**: https://huggingface.co/settings/tokens
2. **Request ImageNet access**: https://huggingface.co/datasets/imagenet-1k  
3. **Set environment**: `export HF_TOKEN="your_token"`
4. **Run validation**: `./start_validation.sh` (AWS) or use Colab notebook

### **✅ Optimization Check**
1. **Check GPU usage**: `./restart_optimized.sh`
2. **If low utilization**: Use `aws_a10g_max_config` or increase batch size
3. **Monitor training**: Look for steady loss decrease and accuracy improvement
4. **Verify logs**: Ensure algorithms are active and no errors occur

### **✅ Production Training**
1. **Validate first**: Always run 1-hour validation before full training
2. **Use optimized config**: `aws_g4dn_12xl_ddp_config` for multi-GPU
3. **Monitor progress**: Set up WandB for experiment tracking
4. **Save checkpoints**: Use `--save-interval 5ep` for regular saves

## 🚀 **Advanced Features**

### **🔥 GPU Optimization Modes**
```bash
# Conservative (safe for any GPU)
python train.py --batch-size 64 --num-workers 2

# Balanced (good performance/safety ratio)  
python train.py --batch-size 128 --num-workers 4 --compile-model

# Aggressive (maximum performance)
python train.py --batch-size 256 --num-workers 8 --compile-model --use-channels-last
```

### **🧪 Experiment Tracking**
```bash
# Basic WandB integration
python train.py --wandb-project "my-imagenet-experiments"

# Advanced tracking with tags
python train.py \
  --wandb-project "resnet50-optimization" \
  --wandb-entity "my-team" \
  --experiment-name "baseline-run-1" \
  --log-interval 50
```

### **⚡ Multi-GPU Training**
```bash
# Data Parallel (single node)
python train.py --batch-size 512  # Automatically uses all available GPUs

# Distributed Data Parallel (recommended)
torchrun --nproc_per_node=4 train.py --batch-size 1024
```

## 📚 **Additional Resources**

- **[AWS Deployment Guide](aws/DEPLOYMENT.md)** - Complete AWS setup and optimization
- **[Project Summary](PROJECT_SUMMARY.md)** - Technical architecture and decisions  
- **[HuggingFace Setup](aws/HUGGINGFACE_SETUP.md)** - Authentication and dataset access
- **[Colab Notebook](colab_validation_1hour.ipynb)** - Interactive training experience

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-optimization`
3. Make your changes with tests
4. Commit: `git commit -m 'Add amazing optimization'`
5. Push: `git push origin feature/amazing-optimization`
6. Create Pull Request

## 📄 **License**

MIT License - see LICENSE file for details.

---

## 🎉 **Ready to Train ImageNet?**

### **🚀 Quick Start Commands:**

```bash
# AWS (Recommended)
./start_validation.sh              # 1-hour validation
./restart_optimized.sh            # Check GPU optimization
./start_training.sh aws_a10g_max_config  # Maximum performance

# Local/Development  
python shared/train.py --use-hf --streaming --data-subset 5000 --epochs 3
```

### **💡 Pro Tips:**
- **Always validate first**: Run 1-hour validation before full training
- **Optimize GPU usage**: Use `./restart_optimized.sh` to maximize performance  
- **Monitor training**: Set up WandB for experiment tracking
- **Save checkpoints**: Regular saves prevent losing progress

**🎯 Achieve 76%+ ImageNet accuracy with optimized, production-ready training!** 🚀