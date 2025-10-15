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
pip install mosaicml>=0.17.0
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
    git clone https://github.com/your-username/mosaic-resnet50.git /opt/mosaic-resnet50 || echo "⚠️ Git clone failed - will need manual setup"
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

# Get config from command line argument or use default
CONFIG_NAME="${1:-aws_g4dn_validation_config}"
echo "🎯 Using configuration: $CONFIG_NAME"

# Training command with config
echo "🚀 Starting ResNet50 training with $CONFIG_NAME..."
python train.py \
    --config configs/training_configs.yaml \
    --config-name $CONFIG_NAME \
    --imagenet-path /mnt/imagenet-data/dataset_hf \
    --checkpoint-dir /opt/checkpoints \
    --log-dir /opt/logs \
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

# Set up environment variables
cat >> /home/ubuntu/.bashrc << 'EOF'
# Training environment
export PATH="/opt/resnet50-env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on instance type
export PYTHONPATH="/opt/mosaic-resnet50:$PYTHONPATH"

# Aliases
alias activate_training='source /home/ubuntu/activate_env.sh'
alias start_training='/home/ubuntu/start_training.sh'
alias check_gpu='nvidia-smi'
alias check_logs='tail -f /opt/logs/training_*.log'
EOF

# Final setup message
cat > /home/ubuntu/README.txt << 'EORM'
Training Instance Setup Complete! 🎉

Quick Start Commands:

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
- ImageNet data: /mnt/imagenet-data

Useful Commands:
- check_gpu: Check GPU status
- check_logs: Monitor training logs
- activate_training: Activate environment and navigate to project

Mount ImageNet Volume First:
   sudo mount /dev/nvme1n1 /mnt/imagenet-data

EORM

echo "🎉 Training instance setup complete!"
echo "📖 See /home/ubuntu/README.txt for next steps"
echo "⏱️ Total setup time: $(date)"
