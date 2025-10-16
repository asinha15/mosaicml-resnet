# 🚀 MosaicML ResNet50 ImageNet Training

A production-ready implementation for training ResNet50 on ImageNet using **MosaicML Composer** with comprehensive optimizations and support for both **Google Colab** and **AWS** environments.

## 📁 **Project Structure**

```
mosaic-resnet/
├── 📁 colab/              # Google Colab environment
│   ├── run_validation.py      # Quick validation runner
│   ├── colab_validation_1hour.py  # 1-hour validation script
│   ├── colab_validation_1hour.ipynb  # Jupyter notebook
│   └── ...colab-specific files
├── 📁 aws/               # AWS production environment  
│   ├── run_validation.py      # Quick validation runner
│   ├── scripts/               # AWS setup scripts
│   └── ...aws-specific files
├── 📁 shared/            # Common training code
│   ├── train.py              # Main training script
│   ├── data_utils.py         # Dataset utilities
│   ├── model.py              # Model definitions
│   └── configs/              # Training configurations
└── README.md             # This file
```

## 🎯 **Quick Start**

### **Option 1: Google Colab (Free GPU)** 🆓

Perfect for experimentation, validation, and learning MosaicML Composer.

#### **1. Setup Colab Environment**
```python
# In Colab cell:
!git clone https://github.com/your-username/mosaic-resnet.git
%cd mosaic-resnet

# Install dependencies
!pip install -r colab/colab_requirements.txt
```

#### **2. Set HuggingFace Token** 
```python
# ImageNet-1k is gated - get token from https://huggingface.co/settings/tokens
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
```

#### **3. Run Validation (5 minutes setup verification)**
```python
%cd colab
!python run_validation.py
```

#### **4. Run 1-Hour Training**
```python
# Fixed script with improved timeout handling and error reporting
!python colab_validation_1hour.py --streaming

# Or use the Jupyter notebook: colab_validation_1hour.ipynb
```

**Expected Output:**
```
🚀 Starting Colab Validation Run (1 hour)
📊 Config: 25k samples, batch_size=128, streaming=True
🎯 Using 6 Composer algorithms: GradientClipping, MixUp, CutMix, LabelSmoothing, EMA, ChannelsLast, BlurPool
  1% 15/195 [01:30<28:45, 0.98ba/s] MulticlassAccuracy/train: 0.1234 
```

---

### **Option 2: AWS Production Environment** ☁️

Optimized for full-scale ImageNet training with powerful GPUs.

#### **1. Launch AWS Instance**
```bash
# Use provided launch script
cd aws
bash launch_training_instance.sh
```

#### **2. Setup Environment (Automated)**
The AWS user-data script automatically:
- Installs Python, PyTorch, MosaicML Composer
- Sets up the training environment
- Handles GPU drivers and dependencies

#### **3. Set HuggingFace Token**
```bash
# SSH into your AWS instance
export HF_TOKEN='your_huggingface_token_here'
echo 'export HF_TOKEN="your_token"' >> ~/.bashrc
```

#### **4. Run Validation (10 minutes setup verification)**
```bash
cd aws
python run_validation.py
```

#### **5. Run Full Production Training**
```bash
cd ../shared

# Full ImageNet training (90 epochs)
python train.py \
  --batch-size 256 \
  --epochs 90 \
  --data-subset full \
  --wandb-project mosaic-resnet50-production \
  --save-folder ./checkpoints

# With streaming (faster startup)
python train.py \
  --batch-size 256 \
  --epochs 90 \
  --data-subset full \
  --streaming \
  --wandb-project mosaic-resnet50-production
```

**Expected Output:**
```
🚀 Starting AWS Validation Run (2 epochs)
📊 Config: 50k samples, batch_size=256, streaming=False  
🎯 Using 7 Composer algorithms: GradientClipping, MixUp, CutMix, RandAugment, LabelSmoothing, EMA, ChannelsLast, BlurPool
  2% 195/9750 [05:30<4:32:20, 0.59ep/s] MulticlassAccuracy/train: 0.4567
```

---

## ⚙️ **Configuration Options**

### **Streaming vs Non-Streaming**

| **Mode** | **Startup Time** | **Memory Usage** | **Deterministic** | **Best For** |
|----------|------------------|------------------|-------------------|--------------|
| **Streaming** (`--streaming`) | Fast (30-90s) | Low | No (sequential) | Colab, quick experiments |
| **Non-streaming** (default) | Slower (5-15min) | Higher | Yes (random sampling) | Production, reproducible training |

### **Data Subset Options**

```bash
--data-subset tiny     # 1,000 samples (quick testing)
--data-subset small    # 10,000 samples (fast validation)  
--data-subset medium   # 50,000 samples (thorough validation)
--data-subset large    # 250,000 samples (partial training)
--data-subset full     # 1,281,167 samples (full ImageNet)
--data-subset 25000    # Custom number
```

### **Composer Algorithms Available**

```bash
# Enable/disable specific algorithms:
--use-mixup            # MixUp data augmentation
--use-cutmix           # CutMix data augmentation  
--use-randaugment      # RandAugment (disabled for streaming)
--use-label-smoothing  # Label smoothing
--use-ema              # Exponential Moving Average
--use-channels-last    # Channels-last memory format
--use-blurpool         # BlurPool anti-aliasing
--use-sam              # Sharpness-Aware Minimization
--use-swa              # Stochastic Weight Averaging
```

---

## 🔧 **Troubleshooting**

### **Training Stops with Errors**

#### **"Training error: 0" (Signal Issue)**
```
❌ Error: Training stopped at 2% with "Training error: 0"
```
**Root Cause:** Signal-based timeout conflicts in Colab environment.
**Solution:** Fixed - now uses polling-based timeout with comprehensive error handling.

#### **KeyError: 0 / Cache Issues (Streaming Dataset)**
```
❌ Error: KeyError: 0 in streaming dataset cache
❌ Error: CACHE BUG: Failed to cache item at index 0
```
**Root Cause:** DataLoader epoch restarts with streaming datasets - requests index 0 when iterator is far ahead.
**Solution:** Fixed - added epoch restart detection and proper iterator reset handling.

**Both issues fixed in updated scripts:**
```python
# In Colab (all fixes applied):
%cd colab  
!python colab_validation_1hour.py --streaming

# In AWS (same fixes now applied):
python aws/run_validation.py
python shared/train.py --streaming --batch-size 256 --epochs 90
```

### **Authentication Error**
```
❌ Error: 401 Unauthorized
```
**Fix:** Get HuggingFace token and set `HF_TOKEN` environment variable.
1. Visit: https://huggingface.co/datasets/imagenet-1k
2. Request access (approval takes 1-2 days)  
3. Get token: https://huggingface.co/settings/tokens
4. Set token: `export HF_TOKEN='your_token'`

### **Shape Mismatch Error**
```
❌ Error: output with shape [1, 224, 224] doesn't match [3, 224, 224]
```
**Fix:** Automatically handled by RGB conversion in `data_utils.py`.

### **Performance Issues**
```
Training stuck at: 0% 0/9750 [00:00<?, ?ba/s]
```
**Fix:** Use streaming mode or reduce `num_workers`:
```bash
python train.py --streaming --num-workers 2
```

### **Memory Issues (Colab)**
```
❌ RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch size:
```bash
python colab_validation_1hour.py --batch-size 64
```

---

## 📊 **Performance Benchmarks**

### **Colab (T4 GPU)**
- **Batch Size:** 128 (optimal for T4)
- **Training Speed:** ~2-5 batches/second
- **Memory Usage:** ~15GB/16GB
- **1-Hour Training:** ~195 batches, 25k samples

### **AWS (g4dn.xlarge)**  
- **Batch Size:** 256 (optimal for T4)
- **Training Speed:** ~3-7 batches/second
- **Memory Usage:** ~14GB/16GB
- **Full Training:** ~90 hours for 90 epochs

### **AWS (g4dn.12xlarge - 4xT4)**
- **Batch Size:** 1024 (256 per GPU)
- **Training Speed:** ~12-25 batches/second  
- **Memory Usage:** ~56GB/64GB total
- **Full Training:** ~20-25 hours for 90 epochs

---

## 🛠️ **Advanced Usage**

### **Custom Configuration**
```bash
# Use predefined configurations
python train.py --config shared/configs/training_configs.yaml --config-name aws_g4dn_12xl_ddp_config

# Override specific parameters
python train.py --batch-size 512 --lr 0.2 --epochs 120
```

### **Learning Rate Finder**
```bash
# Find optimal learning rate
python train.py --find-lr --data-subset small
```

### **Distributed Training (Multi-GPU)**
```bash
# DDP training on multiple GPUs
torchrun --nproc_per_node=4 train.py --batch-size 1024
```

### **WandB Integration**
```bash
# Track experiments with Weights & Biases
python train.py --wandb-project my-project --wandb-group experiment-1
```

---

## 📚 **Documentation**

- **[Colab README](colab/COLAB_README.md)** - Detailed Colab setup and troubleshooting
- **[AWS Training Fixes](aws/AWS_TRAINING_FIXES.md)** - Production fixes applied to AWS scripts
- **[Deployment Guide](aws/DEPLOYMENT.md)** - AWS infrastructure and deployment  
- **[HuggingFace Setup](aws/HUGGINGFACE_SETUP.md)** - Authentication and dataset access
- **[Validation Quickstart](shared/VALIDATION_QUICKSTART.md)** - Quick validation procedures

---

## 🎯 **Expected Results**

### **Validation Runs**
- **Colab Validation:** Should complete without errors, showing steady progress
- **AWS Validation:** Should utilize full GPU, faster than Colab
- **Both:** Should show algorithm application and reasonable loss curves

### **Full Training Results**  
- **Top-1 Accuracy:** ~76-78% (ImageNet validation)
- **Top-5 Accuracy:** ~93-95% (ImageNet validation)
- **Training Time:** 20-90 hours (depending on hardware)
- **Final Loss:** ~0.8-1.2 (cross-entropy)

---

## 🚀 **Getting Started Checklist**

### **For Colab Users:**
- [ ] Clone repository in Colab
- [ ] Install requirements: `pip install -r colab/colab_requirements.txt`  
- [ ] Set HF_TOKEN: `os.environ['HF_TOKEN'] = 'your_token'`
- [ ] Run validation: `python colab/run_validation.py`
- [ ] Start training: `python colab/colab_validation_1hour.py --streaming`

### **For AWS Users:**  
- [ ] Launch AWS instance: `bash aws/launch_training_instance.sh`
- [ ] SSH into instance and set token: `export HF_TOKEN='your_token'`
- [ ] Run validation: `python aws/run_validation.py`
- [ ] Start production training: `python shared/train.py --batch-size 256 --epochs 90 --data-subset full`

---

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 **Acknowledgments**

- **MosaicML** for the excellent Composer library
- **HuggingFace** for the datasets and model hub
- **PyTorch** team for the deep learning framework
- **Google Colab** for free GPU access
- **AWS** for scalable cloud infrastructure

---

**🎉 Ready to train ImageNet? Choose your platform and start training!** 🚀