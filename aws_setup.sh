#!/bin/bash
# AWS g4dn Instance Setup Script for ResNet50 ImageNet Training

set -e  # Exit on any error

echo "🚀 Setting up AWS g4dn instance for ResNet50 training..."

# Update system
sudo apt-get update
sudo apt-get install -y git wget curl htop nvtop

# Install Miniconda
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# Create conda environment
echo "Creating conda environment for ResNet50 training..."
conda create -n resnet50 python=3.10 -y
source $HOME/miniconda3/bin/activate resnet50

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install MosaicML and other requirements
echo "Installing MosaicML Composer and dependencies..."
pip install mosaicml>=0.17.0
pip install datasets>=2.14.0 transformers>=4.30.0
pip install wandb>=0.15.0 torchmetrics>=1.0.0
pip install boto3 awscli
pip install matplotlib seaborn tqdm

# Clone repository (replace with your actual repo)
echo "Setting up project directory..."
if [ ! -d "$HOME/mosaic-resnet50" ]; then
    mkdir -p $HOME/mosaic-resnet50
    cd $HOME/mosaic-resnet50
    
    # You would typically clone from your git repository here
    # git clone https://github.com/yourusername/mosaic-resnet50.git .
    
    echo "Please upload your project files to $HOME/mosaic-resnet50"
fi

# Set up data directory
mkdir -p $HOME/data
mkdir -p $HOME/checkpoints
mkdir -p $HOME/logs

# Configure environment variables
echo "export MOSAIC_RESNET50_HOME=$HOME/mosaic-resnet50" >> ~/.bashrc
echo "export CHECKPOINTS_DIR=$HOME/checkpoints" >> ~/.bashrc
echo "export DATA_DIR=$HOME/data" >> ~/.bashrc

# Install GPU monitoring
pip install gpustat

echo "✅ Setup completed! Please:"
echo "1. Upload your project files to $HOME/mosaic-resnet50"
echo "2. Configure wandb: wandb login"
echo "3. Run source ~/.bashrc to refresh environment"
echo "4. Test with: python -c 'import torch; print(torch.cuda.is_available())'"

# Display system info
echo "📊 System Information:"
nvidia-smi
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
