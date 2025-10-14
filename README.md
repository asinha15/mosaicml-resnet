# MosaicML ResNet50 ImageNet Training

A comprehensive 3-phase implementation for training ResNet50 on ImageNet-1K using MosaicML Composer. Optimized for AWS g4dn instances with efficient ImageNet storage and multi-GPU DDP training.

## 🎯 Multi-Phase Training Strategy

- **Phase 1**: ✅ Colab sanity test with T4 GPU and CIFAR-10/ImageNet subset
- **Phase 2**: AWS g4dn.xlarge validation (1 hour, 25K samples, single T4 GPU)
- **Phase 3**: AWS g4dn.12xlarge production (4x T4 GPUs, full ImageNet, >78% accuracy)

## 📊 ImageNet-1K Dataset Strategy

### 🏆 Recommended: EBS Snapshot Approach

**Best option for repeated training:** Create EBS snapshot once, reuse across instances.

```bash
# 1. Initial setup (once): Create 200GB EBS volume + download ImageNet
./scripts/setup_imagenet.sh setup

# 2. Create reusable snapshot
./scripts/setup_imagenet.sh snapshot
# Outputs: snap-abc123def (save this ID!)

# 3. Future instances: Launch with snapshot-based EBS volume
aws ec2 create-volume --snapshot-id snap-abc123def \
  --availability-zone us-west-2a --size 200 --volume-type gp3

# 4. Mount automatically via user-data script
./scripts/setup_imagenet.sh ebs-only
```

**Benefits:**
- ✅ **Fast instance launches**: ~2 minutes vs 30+ minutes download
- ✅ **Cost effective**: Pay for storage once, reuse unlimited times  
- ✅ **No network dependency**: Data always available locally
- ✅ **Consistent performance**: gp3 EBS provides predictable I/O

### Alternative Options (Less Recommended)

| Method | Setup Time | Cost/Month | Reliability | Best For |
|--------|------------|------------|-------------|----------|
| **EBS Snapshot** | 30 min once | $20-25 | ⭐⭐⭐⭐⭐ | **Recommended** |
| S3 Copy | ~15 min/run | $15-20 + transfer | ⭐⭐⭐⭐ | Occasional use |
| HuggingFace Direct | ~30 min/run | Transfer only | ⭐⭐⭐ | Testing only |

## 🚀 Quick Start Guide

### Phase 1: Colab Validation ✅

```bash
# Already working - run colab_sanity_test.ipynb
# Tests: PyTorch, Composer, data loading, T4 GPU compatibility
```

### Phase 2: AWS Single GPU Validation (1 Hour)

```bash
# 1. Launch g4dn.xlarge instance
# 2. Setup instance and ImageNet
curl -sL https://your-repo/scripts/aws_setup.sh | bash
./scripts/setup_imagenet.sh setup

# 3. Run 1-hour validation training
./train_phase2.sh
```

**Expected Results:**
- Runtime: ~1 hour
- Dataset: 25K samples (~2% ImageNet)  
- Accuracy: ~40-50% (subset validation)
- Cost: ~$0.50 (1 hour g4dn.xlarge)

### Phase 3: AWS Multi-GPU Production (>78% Accuracy)

```bash
# 1. Launch g4dn.12xlarge instance with snapshot-based EBS volume
# 2. Setup instance (ImageNet already available)
curl -sL https://your-repo/scripts/aws_setup.sh | bash
./scripts/setup_imagenet.sh ebs-only  # Just mount existing data

# 3. Run full production training
./train_phase3.sh
```

**Expected Results:**
- Runtime: 12-16 hours
- Dataset: Full ImageNet-1K (1.2M images)
- Accuracy: >78% top-1 (target)
- Hardware: 4x T4 GPUs with DDP
- Cost: ~$60-80 (16 hours g4dn.12xlarge)

### Google Colab (Sanity Test)

1. **Open the notebook**: [`notebooks/colab_sanity_test.ipynb`](notebooks/colab_sanity_test.ipynb)
2. **Run all cells** - the notebook includes:
   - Automatic package installation
   - HuggingFace ImageNet subset loading
   - Quick 2-epoch training test
   - Memory usage analysis

### Local Development

```bash
# Clone and setup
git clone <your-repo>
cd mosaic-resnet50
pip install -r requirements.txt

# Quick test with small subset
python train.py --data-subset tiny --epochs 2 --dry-run

# Learning rate finder
python train.py --find-lr --data-subset small

# Development training
python train.py --data-subset small --epochs 10 --wandb-project dev-test
```

### AWS g4dn Deployment

```bash
# 1. Launch AWS g4dn.xlarge instance with Deep Learning AMI
# 2. SSH into instance and run setup
wget https://raw.githubusercontent.com/yourusername/mosaic-resnet50/main/aws_setup.sh
chmod +x aws_setup.sh
./aws_setup.sh

# 3. Upload project files or clone repository
# 4. Configure wandb
wandb login

# 5. Find optimal learning rate
./aws_train.sh aws_g4dn resnet50-lr-test --find-lr

# 6. Start full training
./aws_train.sh aws_g4dn resnet50-production
```

## 📊 Dataset Integration

### HuggingFace ImageNet-1K

The project uses HuggingFace's `imagenet-1k` dataset for seamless subset loading:

```python
from data_utils import create_dataloaders, get_subset_size

# Available subset sizes
sizes = {
    'tiny': 1000,      # Quick testing
    'small': 10000,    # Development
    'medium': 100000,  # Ablation studies
    'large': 500000,   # Pre-production
    'full': None       # Full ImageNet (1.2M images)
}

train_loader, val_loader = create_dataloaders(
    batch_size=256,
    subset_size=get_subset_size('small'),
    use_hf=True
)
```

### Data Augmentation Pipeline

- **Training**: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **Validation**: Resize + CenterCrop
- **Normalization**: ImageNet statistics
- **Composer Augmentations**: MixUp, CutMix, RandAugment

## 🧠 Model Architecture

### ResNet50 Specifications

- **Parameters**: ~25.6M trainable parameters
- **Architecture**: Standard ResNet50 with bottleneck blocks
- **Initialization**: He initialization + zero-init residual
- **Output**: 1000 classes (ImageNet)

### Composer Integration

```python
from model import create_resnet50_composer

model = create_resnet50_composer(
    num_classes=1000,
    pretrained=False,  # Training from scratch
    compile_model=True  # PyTorch 2.0 compilation
)
```

## ⚡ MosaicML Composer Optimizations

### Enabled Algorithms

| Algorithm | Purpose | Impact |
|-----------|---------|---------|
| **MixUp** | Data augmentation | +1-2% accuracy |
| **CutMix** | Data augmentation | +0.5-1% accuracy |
| **RandAugment** | Advanced augmentation | +0.5% accuracy |
| **Label Smoothing** | Regularization | +0.3-0.5% accuracy |
| **EMA** | Model averaging | +0.2-0.4% accuracy |
| **ChannelsLast** | Memory optimization | 10-15% speedup |
| **BlurPool** | Anti-aliasing | +0.2% accuracy |
| **SWA** | Weight averaging | +0.1-0.3% accuracy |

### Training Configuration

```python
from composer import Trainer
from composer.algorithms import *

algorithms = [
    MixUp(alpha=0.2),
    CutMix(alpha=1.0), 
    RandAugment(severity=9, depth=2),
    LabelSmoothing(smoothing=0.1),
    EMA(half_life='50ba'),
    ChannelsLast(),
    BlurPool(replace_convs=True),
    SWA(swa_start='10ep')
]
```

## 🔧 Configuration Management

### Multi-Phase Configurations

- **`colab_config`**: ✅ T4 GPU, CIFAR-10/tiny subset, 3 epochs  
- **`aws_g4dn_validation_config`**: Single T4, 25K samples, 10 epochs, 1 hour
- **`aws_g4dn_12xl_ddp_config`**: 4x T4 DDP, full ImageNet, 90 epochs, >78% target
- **`dev_config`**: Local development, 10K samples

### ImageNet Storage Configurations

```yaml
# Phase 2 & 3: Use pre-downloaded ImageNet
use_hf: false
imagenet_path: '/opt/imagenet'

# Development: Use HuggingFace streaming  
use_hf: true
data_subset: 'small'
```

See [`configs/training_configs.yaml`](configs/training_configs.yaml) for detailed settings.

## 📈 Performance Expectations

### Phase 2: Validation Results (g4dn.xlarge)

| Metric | Value | Notes |
|--------|-------|-------|
| **Runtime** | ~1 hour | Target validation time |
| **Dataset** | 25K samples | ~2% of ImageNet |
| **Batch Size** | 256 | Single T4 GPU optimized |
| **Accuracy** | ~40-50% | Subset validation benchmark |
| **Cost** | ~$0.50 | 1 hour g4dn.xlarge |

### Phase 3: Production Results (g4dn.12xlarge)

| Metric | Target | Configuration |
|--------|--------|---------------|
| **Accuracy** | **>78%** | Full ImageNet training |
| **Runtime** | 12-16 hours | 4x T4 GPU with DDP |
| **Throughput** | ~2000 samples/sec | Across 4 GPUs |
| **Memory** | ~14GB/GPU | With mixed precision |
| **Cost** | ~$60-80 | Full production run |

## 💡 AWS Infrastructure Best Practices

### Instance Selection

| Phase | Instance | GPUs | Memory | Use Case | Hourly Cost |
|-------|----------|------|--------|----------|-------------|
| Phase 2 | g4dn.xlarge | 1x T4 | 64GB | Validation | ~$0.50 |
| Phase 3 | g4dn.12xlarge | 4x T4 | 192GB | Production | ~$3.90 |

### EBS Volume Strategy

```bash
# Recommended setup for repeated training
1. Create 200GB gp3 EBS volume (better price/performance than gp2)
2. Download ImageNet once: ./scripts/setup_imagenet.sh setup  
3. Create snapshot: ./scripts/setup_imagenet.sh snapshot
4. Future launches: Create volumes from snapshot (2 min vs 30 min)
```

**Cost Comparison (Monthly):**
- EBS gp3 200GB: ~$20/month
- Repeated HF downloads: $30-50+/month in transfer costs
- **Savings**: ~50%+ for multiple training runs

### Multi-GPU DDP Configuration  

```python
# Composer handles DDP automatically
ddp_enabled: true
num_gpus: 4
batch_size: 128  # Per GPU (128 × 4 = 512 effective)
lr: 0.4         # Scaled for larger batch size (0.1 × 4)
```

## 📚 Usage Examples

### Learning Rate Finding

```bash
# Find optimal learning rate
python train.py \
    --find-lr \
    --data-subset small \
    --batch-size 256 \
    --wandb-project lr-finder
```

### Production Training

```bash
# Full ImageNet training
python train.py \
    --experiment-name production-run-1 \
    --data-subset full \
    --batch-size 512 \
    --epochs 90 \
    --lr 0.1 \
    --use-mixup --use-cutmix --use-randaugment \
    --use-label-smoothing --use-ema \
    --use-channels-last --use-blurpool --use-swa \
    --compile-model \
    --wandb-project mosaic-resnet50
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py \
    --checkpoint ./checkpoints/latest-rank0.pt \
    --data-subset full \
    --batch-size 512
```

## 🔍 Monitoring and Debugging

### Weights & Biases Integration

- **Training metrics**: Loss, accuracy, learning rate
- **System metrics**: GPU utilization, memory usage
- **Algorithm metrics**: MixUp lambda, augmentation strength
- **Speed metrics**: Samples/sec, batches/sec

### Key Metrics to Watch

1. **Training Loss**: Should decrease steadily
2. **Validation Accuracy**: Target >78% for production
3. **GPU Memory**: Monitor for OOM errors
4. **Training Speed**: ~1000+ samples/sec on g4dn.xlarge

## 🚨 Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or enable gradient checkpointing
2. **Slow data loading**: Increase `num_workers`, check HF dataset cache
3. **Poor convergence**: Run LR finder, check data augmentation strength
4. **Low accuracy**: Ensure proper normalization, check algorithm configurations

### Debug Commands

```bash
# Test data loading
python -c "from data_utils import create_dataloaders; create_dataloaders(batch_size=32, subset_size=100)"

# Test model forward pass
python -c "from model import create_resnet50_composer; import torch; model = create_resnet50_composer(); x = torch.randn(2, 3, 224, 224); print(model(x).shape)"

# Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
```

## 📋 Requirements

### Core Dependencies

- Python 3.8+
- PyTorch 2.0+
- MosaicML Composer 0.17+
- HuggingFace Datasets 2.14+
- Weights & Biases

### Hardware Requirements

- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory (T4, V100, A10G, A100)
- **CPU**: 8+ cores for data loading
- **RAM**: 32GB+ for full dataset

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: [MosaicML Composer Docs](https://docs.mosaicml.com/)

---

Built with ❤️ using [MosaicML Composer](https://github.com/mosaicml/composer)
