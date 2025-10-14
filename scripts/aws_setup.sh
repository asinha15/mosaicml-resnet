#!/bin/bash
# AWS g4dn Instance Setup Script for Multi-Phase ResNet50 Training

set -e
echo "🚀 AWS g4dn Instance Setup for ResNet50 Training"

# Configuration
PROJECT_DIR="/opt/mosaic-resnet50"
IMAGENET_DIR="/opt/imagenet"
CHECKPOINTS_DIR="/opt/checkpoints"

# Function to detect instance type and GPU configuration
detect_instance() {
    INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
    echo "📡 Detected instance type: $INSTANCE_TYPE"
    
    case $INSTANCE_TYPE in
        "g4dn.xlarge")
            export NUM_GPUS=1
            export PHASE="phase2"
            echo "🎯 Phase 2: Single GPU validation training"
            ;;
        "g4dn.12xlarge")
            export NUM_GPUS=4  
            export PHASE="phase3"
            echo "🎯 Phase 3: Multi-GPU production training"
            ;;
        *)
            echo "⚠️  Unsupported instance type: $INSTANCE_TYPE"
            echo "Supported: g4dn.xlarge (Phase 2), g4dn.12xlarge (Phase 3)"
            exit 1
            ;;
    esac
}

# Function to setup system dependencies
setup_system() {
    echo "🔧 Setting up system dependencies..."
    
    # Update system
    sudo apt-get update
    sudo apt-get install -y git wget curl htop nvtop tree unzip
    
    # Install AWS CLI v2
    if ! command -v aws &> /dev/null; then
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
    
    # Create project directories
    sudo mkdir -p $PROJECT_DIR $CHECKPOINTS_DIR
    sudo chown $USER:$USER $PROJECT_DIR $CHECKPOINTS_DIR
}

# Function to setup Python environment
setup_python() {
    echo "🐍 Setting up Python environment..."
    
    # Install Miniconda if not present
    if [ ! -d "$HOME/miniconda3" ]; then
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $HOME/miniconda3
        rm miniconda.sh
        export PATH="$HOME/miniconda3/bin:$PATH"
        conda init bash
        source ~/.bashrc
    fi
    
    # Create conda environment
    conda create -n resnet50 python=3.10 -y
    source $HOME/miniconda3/bin/activate resnet50
    
    # Install PyTorch with CUDA support for T4 GPUs
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Install MosaicML Composer and dependencies
    pip install mosaicml>=0.17.0 datasets>=2.14.0 transformers>=4.30.0
    pip install wandb>=0.15.0 torchmetrics>=1.0.0 
    pip install matplotlib seaborn tqdm pyyaml
    
    echo "✅ Python environment setup completed!"
}

# Function to setup project code
setup_project() {
    echo "📂 Setting up project code..."
    
    cd $PROJECT_DIR
    
    # Clone or setup project (modify as needed)
    if [ ! -d ".git" ]; then
        # If project not already cloned, you would clone here
        # git clone https://github.com/yourusername/mosaic-resnet50.git .
        echo "📝 Please upload your project files to $PROJECT_DIR"
        echo "   Or clone from your repository"
    fi
    
    # Set up environment variables
    echo "export MOSAIC_RESNET50_HOME=$PROJECT_DIR" >> ~/.bashrc
    echo "export IMAGENET_DIR=$IMAGENET_DIR" >> ~/.bashrc 
    echo "export CHECKPOINTS_DIR=$CHECKPOINTS_DIR" >> ~/.bashrc
    echo "export CUDA_VISIBLE_DEVICES=0,1,2,3" >> ~/.bashrc  # For g4dn.12xlarge
}

# Function to verify GPU setup
verify_gpu() {
    echo "🔍 Verifying GPU setup..."
    
    nvidia-smi
    echo ""
    
    # Test PyTorch CUDA
    source $HOME/miniconda3/bin/activate resnet50
    python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
EOF
}

# Function to setup W&B
setup_wandb() {
    echo "📊 Setting up Weights & Biases..."
    
    source $HOME/miniconda3/bin/activate resnet50
    
    if [ ! -f "$HOME/.netrc" ]; then
        echo "⚠️  W&B not configured. Please run 'wandb login' manually"
        echo "   You can also set WANDB_API_KEY environment variable"
    else
        echo "✅ W&B already configured"
    fi
}

# Function to create phase-specific training scripts
create_training_scripts() {
    echo "📝 Creating training scripts for $PHASE..."
    
    cd $PROJECT_DIR
    
    if [ "$PHASE" == "phase2" ]; then
        # Phase 2: Single GPU validation script
        cat > train_phase2.sh << 'EOF'
#!/bin/bash
# Phase 2: Single GPU Validation Training (1 hour target)

set -e
source $HOME/miniconda3/bin/activate resnet50

echo "🎯 Starting Phase 2: Single GPU Validation Training"
echo "⏱️  Target runtime: 1 hour"
echo "📊 Dataset: 25K ImageNet samples"

# Check ImageNet availability
if [ ! -d "$IMAGENET_DIR/hf_cache" ]; then
    echo "❌ ImageNet not found at $IMAGENET_DIR"
    echo "Please run setup_imagenet.sh first"
    exit 1
fi

# Run validation training
python train.py \
    --config aws_g4dn_validation_config \
    --experiment-name "phase2-validation-$(date +%Y%m%d-%H%M%S)" \
    --imagenet-path $IMAGENET_DIR \
    --save-folder $CHECKPOINTS_DIR \
    --wandb-project mosaic-resnet50-phase2-validation

echo "✅ Phase 2 validation training completed!"
EOF
        chmod +x train_phase2.sh
        
    elif [ "$PHASE" == "phase3" ]; then
        # Phase 3: Multi-GPU production script  
        cat > train_phase3.sh << 'EOF'
#!/bin/bash
# Phase 3: Multi-GPU Production Training (Full ImageNet, >78% accuracy target)

set -e
source $HOME/miniconda3/bin/activate resnet50

echo "🎯 Starting Phase 3: Multi-GPU Production Training"
echo "🚀 Hardware: 4x T4 GPUs with DDP"
echo "📊 Dataset: Full ImageNet-1K (1.2M images)"
echo "🎨 Target: >78% top-1 accuracy"

# Check ImageNet availability
if [ ! -d "$IMAGENET_DIR/hf_cache" ]; then
    echo "❌ ImageNet not found at $IMAGENET_DIR"
    echo "Please run setup_imagenet.sh first"
    exit 1
fi

# Verify all GPUs are available
python -c "import torch; assert torch.cuda.device_count() == 4, f'Expected 4 GPUs, found {torch.cuda.device_count()}'"

# Run multi-GPU training with DDP
torchrun --nproc_per_node=4 train.py \
    --config aws_g4dn_12xl_ddp_config \
    --experiment-name "phase3-production-$(date +%Y%m%d-%H%M%S)" \
    --imagenet-path $IMAGENET_DIR \
    --save-folder $CHECKPOINTS_DIR \
    --wandb-project mosaic-resnet50-phase3-production \
    --ddp-enabled \
    --num-gpus 4

echo "✅ Phase 3 production training completed!"
echo "📈 Check W&B for final accuracy results"
EOF
        chmod +x train_phase3.sh
    fi
    
    echo "✅ Training scripts created!"
}

# Function to display setup summary
display_summary() {
    echo ""
    echo "🎉 AWS g4dn Setup Complete!"
    echo "=========================="
    echo "Instance Type: $INSTANCE_TYPE"
    echo "Training Phase: $PHASE" 
    echo "GPUs Available: $NUM_GPUS"
    echo "Project Directory: $PROJECT_DIR"
    echo "ImageNet Directory: $IMAGENET_DIR"
    echo "Checkpoints Directory: $CHECKPOINTS_DIR"
    echo ""
    echo "🔄 Next Steps:"
    case $PHASE in
        "phase2")
            echo "1. Setup ImageNet: cd $PROJECT_DIR && ./scripts/setup_imagenet.sh"
            echo "2. Upload project files to $PROJECT_DIR"
            echo "3. Configure W&B: conda activate resnet50 && wandb login"
            echo "4. Run validation: ./train_phase2.sh"
            echo "5. Expected runtime: ~1 hour"
            ;;
        "phase3")
            echo "1. Setup ImageNet: cd $PROJECT_DIR && ./scripts/setup_imagenet.sh"  
            echo "2. Upload project files to $PROJECT_DIR"
            echo "3. Configure W&B: conda activate resnet50 && wandb login"
            echo "4. Run production training: ./train_phase3.sh"
            echo "5. Expected runtime: ~12-16 hours for >78% accuracy"
            ;;
    esac
    echo ""
    echo "💡 Tips:"
    echo "   - Monitor training: wandb dashboard or tail -f logs"
    echo "   - Check GPU usage: watch -n 1 nvidia-smi"
    echo "   - Resume from checkpoint if interrupted"
}

# Main execution
main() {
    echo "Starting AWS g4dn setup for ResNet50 training..."
    
    detect_instance
    setup_system
    setup_python
    setup_project
    verify_gpu
    setup_wandb
    create_training_scripts
    display_summary
}

# Run main function
main "$@"
