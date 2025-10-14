#!/bin/bash
# AWS EC2 User Data Script for Phase 3 (g4dn.12xlarge)  
# This script runs automatically when the instance launches

# Log all output for debugging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "🚀 Starting Phase 3 g4dn.12xlarge instance setup..."

# Update system
apt-get update
apt-get install -y git wget curl htop nvtop tree iotop iftop

# Set up directories
mkdir -p /opt/mosaic-resnet50 /opt/checkpoints /opt/logs
chown ubuntu:ubuntu /opt/mosaic-resnet50 /opt/checkpoints /opt/logs

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm -rf aws awscliv2.zip

# Set up Miniconda for ubuntu user
sudo -u ubuntu bash << 'EOSU'
cd /home/ubuntu
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /home/ubuntu/miniconda3
rm miniconda.sh

# Add to PATH
echo 'export PATH="/home/ubuntu/miniconda3/bin:$PATH"' >> /home/ubuntu/.bashrc
export PATH="/home/ubuntu/miniconda3/bin:$PATH"

# Initialize conda
/home/ubuntu/miniconda3/bin/conda init bash

# Create environment
/home/ubuntu/miniconda3/bin/conda create -n resnet50 python=3.10 -y

# Activate and install packages
source /home/ubuntu/miniconda3/bin/activate resnet50
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install mosaicml>=0.17.0 datasets>=2.14.0 transformers>=4.30.0
pip install wandb>=0.15.0 torchmetrics>=1.0.0 
pip install matplotlib seaborn tqdm pyyaml huggingface_hub

EOSU

# Set up environment variables for multi-GPU
cat >> /home/ubuntu/.bashrc << 'EOF'
export MOSAIC_RESNET50_HOME=/opt/mosaic-resnet50
export IMAGENET_DIR=/opt/imagenet
export CHECKPOINTS_DIR=/opt/checkpoints
export LOGS_DIR=/opt/logs
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
EOF

# Configure system for multi-GPU training
# Increase shared memory for multi-GPU data loading
echo 'tmpfs /dev/shm tmpfs defaults,size=32G 0 0' >> /etc/fstab
mount -o remount /dev/shm

# Set up GPU monitoring script
cat > /home/ubuntu/monitor_gpus.sh << 'EOF'
#!/bin/bash
# GPU monitoring script for Phase 3 training
echo "🔍 Phase 3 Multi-GPU Monitoring"
echo "==============================="

watch -n 2 '
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits

echo ""
echo "=== Training Progress (if running) ==="  
if [ -f /opt/checkpoints/latest_log.txt ]; then
    tail -5 /opt/checkpoints/latest_log.txt
else
    echo "No training log found"
fi

echo ""
echo "=== System Resources ==="
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk "{print \$2}" | cut -d% -f1)% usage"
echo "Memory: $(free -h | grep "^Mem:" | awk "{print \$3\"/\"\$2\" (\" \$3/\$2*100 \"%)\"")"
echo "Disk: $(df -h /opt | tail -1 | awk "{print \$3\"/\"\$2\" (\" \$5 \")\"")"
'
EOF
chmod +x /home/ubuntu/monitor_gpus.sh

# Create verification script
cat > /home/ubuntu/verify_setup.sh << 'EOF'
#!/bin/bash
echo "🔍 Phase 3 Setup Verification"
echo "=============================="

source /home/ubuntu/miniconda3/bin/activate resnet50

echo "GPU Status (Should show 4x T4 GPUs):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "PyTorch Multi-GPU Status:"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)')
        
# Test multi-GPU tensor operations
if torch.cuda.device_count() >= 4:
    print('\\nMulti-GPU Test:')
    x = torch.randn(1000, 1000).cuda()
    print(f'  Created tensor on GPU 0: {x.device}')
    
    # Test all GPUs are accessible
    for i in range(4):
        with torch.cuda.device(i):
            y = torch.randn(100, 100).cuda()
            print(f'  GPU {i} accessible: ✓')
    print('  All GPUs accessible for DDP training!')
else:
    print(f'⚠️  Only {torch.cuda.device_count()} GPUs available (expected 4)')
"

echo ""
echo "System Resources:"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"  
echo "Shared Memory: $(df -h /dev/shm | tail -1 | awk '{print $2}')"

echo ""
echo "Next Steps for Phase 3:"
echo "1. Upload/clone your project to /opt/mosaic-resnet50"
echo "2. Mount ImageNet EBS volume: ./scripts/setup_imagenet.sh ebs-only"
echo "3. Configure W&B: wandb login"
echo "4. Run Phase 3 training: ./train_phase3.sh"  
echo "5. Monitor progress: ./monitor_gpus.sh"
echo ""
echo "Expected Phase 3 Results:"
echo "- Training time: 12-16 hours"
echo "- Target accuracy: >78% top-1"
echo "- Throughput: ~2000+ samples/sec across 4 GPUs"
EOF

chmod +x /home/ubuntu/verify_setup.sh
chown ubuntu:ubuntu /home/ubuntu/verify_setup.sh /home/ubuntu/monitor_gpus.sh

# Set up automatic log rotation for training logs
cat > /etc/logrotate.d/mosaic-training << EOF
/opt/logs/*.log /opt/checkpoints/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF

# Create notification that setup is complete
cat > /opt/SETUP_COMPLETE << EOF
Phase 3 (g4dn.12xlarge) User Data Setup Completed
==============================================
Timestamp: $(date)
Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)

Hardware Configuration:
- GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits) x $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
- CPU Cores: $(nproc)
- Memory: $(free -h | grep '^Mem:' | awk '{print $2}')
- Shared Memory: 32GB (configured for multi-GPU data loading)

What was set up:
- System packages optimized for multi-GPU training
- Miniconda with Python 3.10
- PyTorch with CUDA 11.8 support (multi-GPU ready)
- MosaicML Composer with DDP support
- Project directories: /opt/mosaic-resnet50, /opt/checkpoints, /opt/logs
- GPU monitoring tools
- Log rotation for training logs

Next steps for user:
1. SSH into instance: ssh -i your-key.pem ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
2. Run verification: ./verify_setup.sh
3. Mount ImageNet from existing EBS volume/snapshot
4. Start Phase 3 production training

Instance ready for Phase 3 production training with 4x T4 GPUs!
Target: >78% accuracy on full ImageNet-1K dataset
EOF

echo "✅ Phase 3 user-data setup completed successfully!"
echo "Instance is ready for production multi-GPU training."
