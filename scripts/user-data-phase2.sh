#!/bin/bash
# AWS EC2 User Data Script for Phase 2 (g4dn.xlarge)
# This script runs automatically when the instance launches

# Log all output for debugging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "🚀 Starting Phase 2 g4dn.xlarge instance setup..."

# Update system
apt-get update
apt-get install -y git wget curl htop nvtop tree

# Set up directories
mkdir -p /opt/mosaic-resnet50 /opt/checkpoints
chown ubuntu:ubuntu /opt/mosaic-resnet50 /opt/checkpoints

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

# Set up environment variables
cat >> /home/ubuntu/.bashrc << 'EOF'
export MOSAIC_RESNET50_HOME=/opt/mosaic-resnet50
export IMAGENET_DIR=/opt/imagenet
export CHECKPOINTS_DIR=/opt/checkpoints
export CUDA_VISIBLE_DEVICES=0
EOF

# Create quick verification script
cat > /home/ubuntu/verify_setup.sh << 'EOF'
#!/bin/bash
echo "🔍 Phase 2 Setup Verification"
echo "=============================="

source /home/ubuntu/miniconda3/bin/activate resnet50

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "PyTorch Status:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Next Steps:"
echo "1. Upload/clone your project to /opt/mosaic-resnet50"
echo "2. Set HUGGINGFACE_TOKEN: export HUGGINGFACE_TOKEN='your-token'"  
echo "3. Run: cd /opt/mosaic-resnet50 && ./scripts/setup_imagenet.sh setup"
echo "4. Configure W&B: wandb login"
echo "5. Run Phase 2 training: ./train_phase2.sh"
EOF

chmod +x /home/ubuntu/verify_setup.sh
chown ubuntu:ubuntu /home/ubuntu/verify_setup.sh

# Create notification that setup is complete
cat > /opt/SETUP_COMPLETE << EOF
Phase 2 (g4dn.xlarge) User Data Setup Completed
=============================================
Timestamp: $(date)
Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)

What was set up:
- System packages (git, htop, nvtop, etc.)
- Miniconda with Python 3.10
- PyTorch with CUDA 11.8 support
- MosaicML Composer and dependencies  
- Project directories: /opt/mosaic-resnet50, /opt/checkpoints

Next steps for user:
1. SSH into instance: ssh -i your-key.pem ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
2. Run verification: ./verify_setup.sh
3. Set up ImageNet with HF token
4. Start Phase 2 training

Instance ready for Phase 2 validation training!
EOF

echo "✅ Phase 2 user-data setup completed successfully!"
echo "Instance is ready for Phase 2 training."
