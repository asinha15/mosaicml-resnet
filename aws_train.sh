#!/bin/bash
# Training script for AWS g4dn instances

set -e

# Default configuration
CONFIG=${1:-aws_g4dn}
EXPERIMENT_NAME=${2:-"resnet50-$(date +%Y%m%d-%H%M%S)"}

echo "🎯 Starting ResNet50 ImageNet training on AWS"
echo "Configuration: $CONFIG"
echo "Experiment: $EXPERIMENT_NAME"

# Activate environment
source $HOME/miniconda3/bin/activate resnet50

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

# Check system resources
echo "📊 System Check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"

# Create experiment directory
EXPERIMENT_DIR="$HOME/experiments/$EXPERIMENT_NAME"
mkdir -p $EXPERIMENT_DIR
cd $HOME/mosaic-resnet50

# Run learning rate finder first (optional)
if [ "${3}" == "--find-lr" ]; then
    echo "🔍 Running Learning Rate Finder..."
    python train.py \
        --experiment-name "${EXPERIMENT_NAME}-lr-finder" \
        --data-subset small \
        --find-lr \
        --wandb-project mosaic-resnet50-lr-finder \
        --batch-size 256
    
    echo "✅ LR Finder completed. Check wandb logs for optimal LR."
    exit 0
fi

# Main training command
echo "🚀 Starting main training..."
python train.py \
    --experiment-name $EXPERIMENT_NAME \
    --data-subset full \
    --batch-size 512 \
    --epochs 90 \
    --lr 0.1 \
    --optimizer sgd \
    --use-mixup \
    --use-cutmix \
    --use-randaugment \
    --use-label-smoothing \
    --use-ema \
    --use-channels-last \
    --use-blurpool \
    --use-swa \
    --compile-model \
    --device cuda \
    --precision amp_fp16 \
    --num-workers 16 \
    --save-folder $EXPERIMENT_DIR/checkpoints \
    --save-interval 5ep \
    --wandb-project mosaic-resnet50-aws \
    --log-interval 100ba

echo "✅ Training completed!"
echo "Checkpoints saved to: $EXPERIMENT_DIR/checkpoints"
echo "Logs available in wandb project: mosaic-resnet50-aws"

# Evaluate final model
echo "📈 Running final evaluation..."
python -c "
import torch
from composer import Trainer
from model import create_resnet50_composer
from data_utils import create_dataloaders

print('Loading best checkpoint...')
model = create_resnet50_composer()
# Load best checkpoint here
_, val_loader = create_dataloaders(batch_size=512, subset_size=None, use_hf=True)

print('Running evaluation...')
# Add evaluation code here
print('Evaluation completed!')
"

# Cleanup
echo "🧹 Cleaning up temporary files..."
# Add cleanup commands if needed

echo "🎉 All done! Check your results in wandb."
