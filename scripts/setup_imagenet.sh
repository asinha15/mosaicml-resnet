#!/bin/bash
# ImageNet-1K Download and Setup Script for AWS
# Creates an EBS volume with ImageNet data for fast, reusable access

set -e
echo "🗂️  ImageNet-1K Dataset Setup for AWS Training"

# Configuration
IMAGENET_DIR="/opt/imagenet"
EBS_VOLUME_SIZE="200"  # GB - ImageNet needs ~150GB
EBS_DEVICE="/dev/nvme1n1"  # Typical for additional EBS volume

# Function to download and setup ImageNet
setup_imagenet() {
    echo "📥 Setting up ImageNet-1K dataset..."
    
    # Check for HuggingFace token
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "🔐 HuggingFace token required for ImageNet-1k access"
        echo "Please set HUGGINGFACE_TOKEN environment variable or provide it interactively"
        
        # Try to read from HF cache first
        HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
        if [ -f "$HF_TOKEN_FILE" ]; then
            echo "Found existing HF token file"
            export HUGGINGFACE_TOKEN=$(cat "$HF_TOKEN_FILE")
        else
            # Prompt for token if not found
            read -p "Enter your HuggingFace token (from https://huggingface.co/settings/tokens): " HUGGINGFACE_TOKEN
            
            # Optionally save token for future use
            read -p "Save token for future use? (y/n): " SAVE_TOKEN
            if [ "$SAVE_TOKEN" = "y" ]; then
                mkdir -p "$(dirname "$HF_TOKEN_FILE")"
                echo "$HUGGINGFACE_TOKEN" > "$HF_TOKEN_FILE"
                chmod 600 "$HF_TOKEN_FILE"
                echo "Token saved to $HF_TOKEN_FILE"
            fi
        fi
        
        if [ -z "$HUGGINGFACE_TOKEN" ]; then
            echo "❌ No HuggingFace token provided. Cannot access ImageNet-1k dataset."
            echo "Please visit https://huggingface.co/datasets/imagenet-1k to request access"
            echo "Then create a token at https://huggingface.co/settings/tokens"
            return 1
        fi
    fi
    
    # Create directory
    sudo mkdir -p $IMAGENET_DIR
    sudo chown $USER:$USER $IMAGENET_DIR
    
    # Install HF datasets CLI and login
    echo "🔧 Installing HuggingFace datasets..."
    pip install datasets[cli] huggingface_hub
    
    # Login with token
    echo "🔐 Authenticating with HuggingFace..."
    huggingface-cli login --token $HUGGINGFACE_TOKEN
    
    # Download ImageNet using HF datasets
    echo "📦 Downloading ImageNet-1k dataset (this may take 30-60 minutes)..."
    cd $IMAGENET_DIR
    
    python3 << EOF
import os
import sys
from datasets import load_dataset
from pathlib import Path
from huggingface_hub import login

# Authenticate
try:
    login(token="$HUGGINGFACE_TOKEN")
    print("✅ HuggingFace authentication successful")
except Exception as e:
    print(f"❌ HuggingFace authentication failed: {e}")
    print("Please check your token and ensure you have access to imagenet-1k dataset")
    sys.exit(1)

try:
    print("📥 Downloading ImageNet train split (this is the large one)...")
    train_dataset = load_dataset(
        "imagenet-1k", 
        split="train", 
        cache_dir="$IMAGENET_DIR/hf_cache",
        trust_remote_code=False
    )
    print(f"✅ Train split downloaded: {len(train_dataset):,} samples")

    print("📥 Downloading ImageNet validation split...")  
    val_dataset = load_dataset(
        "imagenet-1k", 
        split="validation", 
        cache_dir="$IMAGENET_DIR/hf_cache",
        trust_remote_code=False
    )
    print(f"✅ Validation split downloaded: {len(val_dataset):,} samples")

    # Create symlinks to standard ImageNet structure for compatibility
    print("🔗 Creating standard ImageNet directory structure...")
    cache_dir = Path("$IMAGENET_DIR/hf_cache")
    train_dir = Path("$IMAGENET_DIR/train")  
    val_dir = Path("$IMAGENET_DIR/val")
    
    # Create directories if they don't exist
    train_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)
    
    # Create info file for easy reference
    info_file = Path("$IMAGENET_DIR/dataset_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"ImageNet-1k Dataset Information\\n")
        f.write(f"================================\\n")
        f.write(f"Downloaded: {train_dataset.info.download_date if hasattr(train_dataset.info, 'download_date') else 'N/A'}\\n")
        f.write(f"Train samples: {len(train_dataset):,}\\n")
        f.write(f"Validation samples: {len(val_dataset):,}\\n")
        f.write(f"Cache location: {cache_dir}\\n")
        f.write(f"\\nTo use in training:\\n")
        f.write(f"  --imagenet-path $IMAGENET_DIR\\n")
        f.write(f"  --use-hf false\\n")
    
    print(f"✅ Dataset info saved to {info_file}")
    print("📂 HuggingFace cache contains the actual data")
    print("🔗 Standard directories created for compatibility")
    
except Exception as e:
    print(f"❌ Failed to download ImageNet: {e}")
    print("\\nCommon issues:")
    print("1. No access to imagenet-1k dataset - request access at https://huggingface.co/datasets/imagenet-1k")
    print("2. Invalid or expired token - create new token at https://huggingface.co/settings/tokens")  
    print("3. Network issues - try again later")
    sys.exit(1)

EOF

    if [ $? -eq 0 ]; then
        echo "✅ ImageNet dataset setup completed!"
        echo "📊 Dataset information:"
        du -sh $IMAGENET_DIR/*
        ls -la $IMAGENET_DIR/
        
        if [ -f "$IMAGENET_DIR/dataset_info.txt" ]; then
            echo ""
            cat "$IMAGENET_DIR/dataset_info.txt"
        fi
    else
        echo "❌ ImageNet setup failed"
        return 1
    fi
}

# Function to create and format EBS volume  
setup_ebs_volume() {
    echo "💾 Setting up EBS volume for ImageNet storage..."
    
    # Check if volume is already attached
    if lsblk | grep -q nvme1n1; then
        echo "EBS volume already attached at $EBS_DEVICE"
        
        # Check if already formatted and mounted
        if mount | grep -q $IMAGENET_DIR; then
            echo "Volume already mounted at $IMAGENET_DIR"
            return 0
        fi
        
        # Format if not already formatted
        if ! blkid $EBS_DEVICE; then
            echo "Formatting EBS volume..."
            sudo mkfs.ext4 $EBS_DEVICE
        fi
        
        # Mount the volume
        echo "Mounting EBS volume..."
        sudo mkdir -p $IMAGENET_DIR
        sudo mount $EBS_DEVICE $IMAGENET_DIR
        sudo chown $USER:$USER $IMAGENET_DIR
        
        # Add to fstab for automatic mounting
        echo "$EBS_DEVICE $IMAGENET_DIR ext4 defaults 0 2" | sudo tee -a /etc/fstab
        
        echo "✅ EBS volume setup completed!"
    else
        echo "⚠️  No additional EBS volume detected."
        echo "Please attach a ${EBS_VOLUME_SIZE}GB EBS volume to this instance first."
        echo "Then run this script again."
        return 1
    fi
}

# Function to create EBS snapshot for reuse
create_ebs_snapshot() {
    echo "📸 Creating EBS snapshot for future use..."
    
    # Get volume ID
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    VOLUME_ID=$(aws ec2 describe-volumes \
        --filters "Name=attachment.instance-id,Values=$INSTANCE_ID" "Name=attachment.device,Values=/dev/sdf" \
        --query "Volumes[0].VolumeId" --output text)
    
    if [ "$VOLUME_ID" != "None" ] && [ "$VOLUME_ID" != "" ]; then
        echo "Creating snapshot of volume $VOLUME_ID..."
        SNAPSHOT_ID=$(aws ec2 create-snapshot \
            --volume-id $VOLUME_ID \
            --description "ImageNet-1K dataset for ResNet50 training $(date +%Y-%m-%d)" \
            --query "SnapshotId" --output text)
        
        echo "✅ Snapshot created: $SNAPSHOT_ID"
        echo "💡 Save this snapshot ID for future instances:"
        echo "   aws ec2 create-volume --snapshot-id $SNAPSHOT_ID --availability-zone \$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone) --size $EBS_VOLUME_SIZE --volume-type gp3"
    else
        echo "⚠️  Could not find EBS volume to snapshot"
    fi
}

# Main execution
case "${1:-setup}" in
    "setup")
        echo "🚀 Full ImageNet setup with EBS volume..."
        setup_ebs_volume
        setup_imagenet
        ;;
    "ebs-only")  
        echo "💾 Setting up EBS volume only..."
        setup_ebs_volume
        ;;
    "download-only")
        echo "📥 Downloading ImageNet to existing directory..."
        setup_imagenet
        ;;
    "snapshot")
        echo "📸 Creating EBS snapshot..."
        create_ebs_snapshot
        ;;
    "restore")
        if [ -z "$2" ]; then
            echo "❌ Usage: $0 restore <snapshot-id>"
            exit 1
        fi
        echo "♻️  Restoring from snapshot $2..."
        # This would typically be done when creating the instance
        echo "To restore, create EBS volume from snapshot when launching instance:"
        echo "aws ec2 create-volume --snapshot-id $2 --availability-zone \$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone) --size $EBS_VOLUME_SIZE --volume-type gp3"
        ;;
    *)
        echo "Usage: $0 [setup|ebs-only|download-only|snapshot|restore <snapshot-id>]"
        echo ""
        echo "Commands:"
        echo "  setup         - Full setup: EBS volume + ImageNet download"
        echo "  ebs-only      - Setup EBS volume only"  
        echo "  download-only - Download ImageNet to existing directory"
        echo "  snapshot      - Create EBS snapshot for reuse"
        echo "  restore       - Instructions for restoring from snapshot"
        exit 1
        ;;
esac

echo ""
echo "📋 Next steps:"
echo "1. Verify dataset: ls -la $IMAGENET_DIR"
echo "2. Test loading: python -c \"from datasets import load_dataset; ds = load_dataset('imagenet-1k', cache_dir='$IMAGENET_DIR/hf_cache', split='train[:100]'); print(f'Loaded {len(ds)} samples')\""
echo "3. For future instances, use: $0 restore <snapshot-id>"
echo ""
echo "💡 EBS Best Practices:"
echo "   - Use gp3 volumes for better price/performance"
echo "   - Create snapshot after setup for quick instance launches"
echo "   - Mount snapshot-based volumes automatically via user-data script"
