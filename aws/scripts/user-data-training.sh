#!/bin/bash
# AWS EC2 User Data Script for Training (g4dn spot instances)
# This script sets up training environment with pip (no conda issues)

# Log all output for debugging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "🚀 Starting g4dn training instance setup..."

# Check available resources
echo "System resources at start:"
free -h
df -h
nvidia-smi
echo "---"

# Update system and install essentials
apt-get update
apt-get install -y python3 python3-pip python3-venv git wget curl htop nvtop tree

# Install NVIDIA drivers if not present (usually pre-installed on Deep Learning AMI)
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Installing NVIDIA drivers..."
    apt-get install -y nvidia-driver-470
fi

# Create Python virtual environment (more reliable than conda)
echo "🐍 Setting up Python environment with venv..."
python3 -m venv /opt/resnet50-env
source /opt/resnet50-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (pip is faster and more reliable)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install MosaicML and other packages
echo "🎼 Installing MosaicML Composer and dependencies..."
# Use version range to avoid metadata issues with 0.12.0, 0.11.1, 0.11.0
pip install "mosaicml>=0.17.0,<=0.18.2" --no-warn-conflicts
pip install datasets>=2.14.0 transformers>=4.30.0 huggingface_hub>=0.17.0
pip install wandb>=0.15.0 torchmetrics>=1.0.0
pip install matplotlib seaborn tqdm pyyaml Pillow

# Test PyTorch installation
echo "🧪 Testing PyTorch installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"

# Set up project directory
echo "📁 Setting up project structure..."
mkdir -p /opt/mosaic-resnet50
mkdir -p /opt/checkpoints
mkdir -p /opt/logs

# Clone project (if not already present)
if [ ! -d "/opt/mosaic-resnet50/.git" ]; then
    git clone https://github.com/your-username/mosaicml-resnet.git /opt/mosaic-resnet50 || echo "⚠️ Git clone failed - will need manual setup"
fi

# Set permissions
chown -R ubuntu:ubuntu /opt/mosaic-resnet50
chown -R ubuntu:ubuntu /opt/checkpoints
chown -R ubuntu:ubuntu /opt/logs

# Create activation script
cat > /home/ubuntu/activate_env.sh << 'EOAE'
#!/bin/bash
# Activate the training environment
source /opt/resnet50-env/bin/activate
cd /opt/mosaic-resnet50
echo "🎯 Training environment activated!"
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "🚀 Ready to start training!"
EOAE

chmod +x /home/ubuntu/activate_env.sh
chown ubuntu:ubuntu /home/ubuntu/activate_env.sh

# Create training start script
cat > /home/ubuntu/start_training.sh << 'EOTS'
#!/bin/bash
set -e

# Activate environment
source /opt/resnet50-env/bin/activate
cd /opt/mosaic-resnet50

# Check if ImageNet data is available
if [ ! -d "/mnt/imagenet-data" ]; then
    echo "❌ ImageNet data not found at /mnt/imagenet-data"
    echo "Please ensure the ImageNet EBS volume is attached and mounted"
    exit 1
fi

echo "✅ ImageNet data found at /mnt/imagenet-data"
ls -la /mnt/imagenet-data/

# Set environment variables for local HuggingFace cache
export HF_HOME="/mnt/imagenet-data/hf_home"
export HF_HUB_CACHE="/mnt/imagenet-data/hf_home/hub"
export TRANSFORMERS_CACHE="/mnt/imagenet-data/transformers_cache"
# Note: HF_DATASETS_OFFLINE will be managed automatically by the training code
echo "🔧 Set HF_HOME=/mnt/imagenet-data/hf_home"
echo "🔧 HuggingFace cache configured for local ImageNet data"

# Get config from command line argument or use default
CONFIG_NAME="${1:-aws_g4dn_validation_config}"
echo "🎯 Using configuration: $CONFIG_NAME"

# Set training parameters based on config
case $CONFIG_NAME in
    "aws_g4dn_validation_config")
        echo "📊 1-Hour Validation Configuration (Local HF Cache)"
        ARGS="--model-type torchvision \
              --compile-model \
              --data-subset 25000 \
              --batch-size 256 \
              --image-size 224 \
              --num-workers 8 \
              --use-hf \
              --epochs 10 \
              --lr 0.05 \
              --weight-decay 1e-4 \
              --momentum 0.9 \
              --optimizer sgd \
              --use-mixup \
              --use-cutmix \
              --use-randaugment \
              --use-label-smoothing \
              --use-ema \
              --use-channels-last \
              --use-blurpool \
              --device auto \
              --precision amp_fp16 \
              --save-interval 2ep \
              --wandb-project mosaic-resnet50-phase2-validation \
              --log-interval 50ba"
        ;;
    "aws_g4dn_12xl_ddp_config")
        echo "📊 Full Training Configuration (4x T4, Local HF Cache)"
        ARGS="--model-type torchvision \
              --compile-model \
              --batch-size 128 \
              --image-size 224 \
              --num-workers 16 \
              --use-hf \
              --epochs 90 \
              --lr 0.4 \
              --weight-decay 1e-4 \
              --momentum 0.9 \
              --optimizer sgd \
              --use-mixup \
              --use-cutmix \
              --use-randaugment \
              --use-label-smoothing \
              --use-ema \
              --use-channels-last \
              --use-blurpool \
              --use-sam \
              --use-swa \
              --device auto \
              --precision amp_fp16 \
              --save-interval 5ep \
              --wandb-project mosaic-resnet50-phase3-production \
              --log-interval 100ba"
        ;;
    "colab_config")
        echo "📊 Colab Test Configuration"
        ARGS="--model-type torchvision \
              --data-subset 1000 \
              --batch-size 64 \
              --image-size 224 \
              --num-workers 2 \
              --use-hf \
              --epochs 3 \
              --lr 0.01 \
              --weight-decay 1e-4 \
              --momentum 0.9 \
              --optimizer sgd \
              --use-mixup \
              --use-label-smoothing \
              --use-channels-last \
              --device auto \
              --precision amp_fp16 \
              --wandb-project mosaic-resnet50-colab \
              --log-interval 10ba"
        ;;
    *)
        echo "❌ Unknown configuration: $CONFIG_NAME"
        echo "Available configs: aws_g4dn_validation_config, aws_g4dn_12xl_ddp_config, colab_config"
        exit 1
        ;;
esac

# Training command
echo "🚀 Starting ResNet50 training with $CONFIG_NAME..."
python ./shared/train.py $ARGS \
    --save-folder /opt/checkpoints \
    --experiment-name ${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S) \
    2>&1 | tee /opt/logs/training_${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log

EOTS

chmod +x /home/ubuntu/start_training.sh
chown ubuntu:ubuntu /home/ubuntu/start_training.sh

# Create validation-specific script for 1-hour test
cat > /home/ubuntu/start_validation.sh << 'EOVS'
#!/bin/bash
set -e

echo "🚀 Starting 1-Hour Validation Test..."
echo "📊 Configuration: aws_g4dn_validation_config"
echo "⏱️  Target duration: ~1 hour"
echo "📦 Dataset: 25K samples (2% of ImageNet)"

# Run the training with validation config
./start_training.sh aws_g4dn_validation_config

echo "✅ 1-Hour validation test completed!"

EOVS

chmod +x /home/ubuntu/start_validation.sh
chown ubuntu:ubuntu /home/ubuntu/start_validation.sh

# Create data validation script
cat > /home/ubuntu/validate_imagenet.sh << 'EOVD'
#!/bin/bash
set -e

echo "🔍 Validating ImageNet Data Setup..."

# Activate environment
source /opt/resnet50-env/bin/activate
cd /opt/mosaic-resnet50

# Set environment variables for HuggingFace cache
export HF_HOME="/mnt/imagenet-data/hf_home"
export HF_HUB_CACHE="/mnt/imagenet-data/hf_home/hub"
export TRANSFORMERS_CACHE="/mnt/imagenet-data/transformers_cache"
# Note: HF_DATASETS_OFFLINE managed automatically by training code

# Check if ImageNet data is mounted
if [ ! -d "/mnt/imagenet-data" ]; then
    echo "❌ ImageNet data not found at /mnt/imagenet-data"
    exit 1
fi

echo "✅ ImageNet data directory found"

# Check for expected HuggingFace cache structure
if [ ! -d "/mnt/imagenet-data/hf_home" ]; then
    echo "❌ Expected HuggingFace cache structure not found"
    echo "   Expected: /mnt/imagenet-data/hf_home"
    echo "   Found:"
    ls -la /mnt/imagenet-data/ | head -10
    exit 1
fi

# Check for ImageNet dataset in HF cache
DATASET_PATH="/mnt/imagenet-data/hf_home/hub/datasets--imagenet-1k"
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ImageNet-1k dataset not found in HF cache"
    echo "   Expected: $DATASET_PATH"
    exit 1
fi

echo "✅ HuggingFace ImageNet cache structure is correct"

# Check cache size
CACHE_SIZE=$(du -sh /mnt/imagenet-data/hf_home | cut -f1)
echo "📊 Dataset statistics:"
echo "   HF Cache size: $CACHE_SIZE"
echo "   Cache location: /mnt/imagenet-data/hf_home"

# Test data loading with Python
echo "🧪 Testing Python data loading..."
python3 -c "
import os
# Set HuggingFace environment variables
os.environ['HF_HOME'] = '/mnt/imagenet-data/hf_home'
os.environ['HF_HUB_CACHE'] = '/mnt/imagenet-data/hf_home/hub'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/imagenet-data/transformers_cache'
# Note: HF_DATASETS_OFFLINE managed by data_utils automatically

try:
    from shared.data_utils import create_dataloaders
    print('   ✅ Successfully imported data_utils')
    
    # Test creating dataloaders with HuggingFace (local cache)
    # Provide dummy token for cached datasets (sometimes required)
    import os
    dummy_token = os.environ.get('HF_TOKEN', 'dummy_token_for_cache')
    
    train_loader, val_loader = create_dataloaders(
        batch_size=8,
        num_workers=1, 
        subset_size=100,
        use_hf=True,  # Use HuggingFace dataset
        streaming=False,  # Use local cache, not streaming
        token=dummy_token
    )
    print(f'   ✅ Created train loader: {len(train_loader)} batches')
    print(f'   ✅ Created val loader: {len(val_loader)} batches')
    
    # Test loading one batch
    batch = next(iter(train_loader))
    images, labels = batch
    print(f'   ✅ Loaded batch: images {images.shape}, labels {labels.shape}')
    print(f'   ✅ Label range: {labels.min().item()}-{labels.max().item()}')
    
except Exception as e:
    print(f'   ❌ Data loading test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo "✅ HuggingFace cache validation completed successfully!"
echo "🚀 Ready for training with local HuggingFace ImageNet cache"

EOVD

chmod +x /home/ubuntu/validate_imagenet.sh
chown ubuntu:ubuntu /home/ubuntu/validate_imagenet.sh

# Set up environment variables
cat >> /home/ubuntu/.bashrc << 'EOF'
# Training environment
export PATH="/opt/resnet50-env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on instance type
export PYTHONPATH="/opt/mosaic-resnet50:$PYTHONPATH"

# HuggingFace cache paths (local ImageNet data)
export HF_HOME="/mnt/imagenet-data/hf_home"
export HF_HUB_CACHE="/mnt/imagenet-data/hf_home/hub"
export TRANSFORMERS_CACHE="/mnt/imagenet-data/transformers_cache"
# Note: HF_DATASETS_OFFLINE managed automatically by training code

# Aliases
alias activate_training='source /home/ubuntu/activate_env.sh'
alias start_training='/home/ubuntu/start_training.sh'
alias validate_data='/home/ubuntu/validate_imagenet.sh'
alias check_gpu='nvidia-smi'
alias check_logs='tail -f /opt/logs/training_*.log'

# To use HuggingFace streaming from internet instead of local cache:
# Just add --streaming flag (offline mode handled automatically)
# Example: python ./shared/train.py --use-hf --streaming --other-args
EOF

# Final setup message
cat > /home/ubuntu/README.txt << 'EORM'
Training Instance Setup Complete! 🎉

Quick Start Commands:

🔍 Validate Data Setup (RUN FIRST):
   ./validate_imagenet.sh

🚀 1-Hour Validation Test (RECOMMENDED):
   ./start_validation.sh

🎯 Custom Configuration Training:
   ./start_training.sh [config_name]

Available Configurations:
- aws_g4dn_validation_config  (1-hour test, 25K samples)
- aws_g4dn_12xl_ddp_config   (full training, 4x GPU)
- colab_config               (tiny test, 1K samples)

Examples:
   ./start_training.sh aws_g4dn_validation_config  # 1-hour validation
   ./start_training.sh aws_g4dn_12xl_ddp_config    # full training

Environment Details:
- Python virtual environment: /opt/resnet50-env
- Project directory: /opt/mosaic-resnet50
- Checkpoints: /opt/checkpoints
- Logs: /opt/logs
- ImageNet data: /mnt/imagenet-data/hf_home (LOCAL HuggingFace Cache)

Data Configuration:
💾 Default: Uses LOCAL HuggingFace cache from /mnt/imagenet-data/hf_home
   - Automatically detects and uses local cache when available
   - --use-hf flag enabled (uses HuggingFace dataset format)
   - No --streaming flag (uses local cache, not streaming)
   - Smart offline/online mode switching as needed

🌊 Alternative: To stream from internet instead:
   python ./shared/train.py --use-hf --streaming --other-args

Useful Commands:
- validate_data: Validate HuggingFace cache setup
- check_gpu: Check GPU status  
- check_logs: Monitor training logs
- activate_training: Activate environment and navigate to project

Mount ImageNet Volume First:
   sudo mkdir -p /mnt/imagenet-data
   sudo mount /dev/sdf /mnt/imagenet-data
   # Alternative: sudo mount /dev/sdf1 /mnt/imagenet-data

EORM

echo "🎉 Training instance setup complete!"
echo "📖 See /home/ubuntu/README.txt for next steps"
echo "⏱️ Total setup time: $(date)"
