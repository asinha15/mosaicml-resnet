# MosaicML ResNet50 ImageNet Training

A comprehensive implementation for training ResNet50 on ImageNet-1K using MosaicML Composer with state-of-the-art optimizations. Designed to achieve >78% accuracy on AWS g4dn instances.

## 🎯 Project Goals

- **Phase 1**: Colab sanity test with T4 GPU and ImageNet subset
- **Phase 2**: Full-scale training preparation with Composer optimizations
- **Phase 3**: AWS g4dn deployment targeting >78% top-1 accuracy

## 🚀 Quick Start

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

### Pre-defined Configurations

- **`colab_config`**: T4 GPU, small subset, 3 epochs
- **`dev_config`**: Local development, 10k samples
- **`aws_g4dn_config`**: Production training, full dataset
- **`aws_g4dn_2xl_config`**: Multi-GPU training

### Configuration Files

See [`configs/training_configs.yaml`](configs/training_configs.yaml) for detailed settings.

## 📈 Expected Performance

### Accuracy Targets

| Configuration | Hardware | Dataset Size | Expected Top-1 | Training Time |
|---------------|----------|--------------|-----------------|---------------|
| Colab Test | T4 | 1K samples | ~5-10% | 10 minutes |
| Development | Local GPU | 10K samples | ~15-25% | 1 hour |
| AWS g4dn.xlarge | V100/A10G | Full ImageNet | **>78%** | 12-16 hours |
| AWS g4dn.2xlarge | 2x GPU | Full ImageNet | **>79%** | 6-8 hours |

### Memory Requirements

- **Colab T4 (16GB)**: Batch size 64, works reliably
- **AWS g4dn.xlarge (16GB)**: Batch size 512 with mixed precision
- **Local GPU (8GB)**: Batch size 128-256 depending on model

## 🛠 Development Workflow

### Phase 1: Colab Sanity Test ✅

- [x] Basic setup and dependencies
- [x] HuggingFace dataset integration
- [x] Composer model implementation
- [x] T4 GPU memory optimization
- [x] Quick training validation

### Phase 2: Full Training Setup

- [ ] Learning rate finder integration
- [ ] Advanced Composer algorithms
- [ ] Checkpointing and resuming
- [ ] Comprehensive logging
- [ ] AWS deployment scripts

### Phase 3: Production Optimization

- [ ] Multi-GPU support (if using g4dn.2xlarge)
- [ ] Hyperparameter optimization
- [ ] >78% accuracy validation
- [ ] Cost optimization analysis

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
