#!/bin/bash
# AWS EC2 User Data Script for ImageNet Download (c5n.xlarge)
# This script downloads ImageNet-1k dataset quickly and creates an EBS snapshot
#
# Usage: Pass HuggingFace token as user-data or set HF_TOKEN environment variable
# Example: --user-data "$(cat user-data-download.sh | sed 's/HF_TOKEN_PLACEHOLDER/your_actual_token/')"

# Extract HF token from user-data if provided as parameter
HF_TOKEN="${HF_TOKEN:-HF_TOKEN_PLACEHOLDER}"

# Log all output for debugging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "🚀 Starting ImageNet download instance setup (c5n.xlarge)..."

# Check available resources
echo "System resources at start:"
free -h
df -h
echo "Network interfaces:"
ip addr show
echo "---"

# Update system and install essential packages
apt-get update
apt-get install -y python3 python3-pip python3-venv git wget curl htop tree

# Create Python virtual environment for isolated package management
echo "🐍 Setting up Python virtual environment..."
python3 -m venv /opt/imagenet-env
source /opt/imagenet-env/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install Python packages in virtual environment (faster and more reliable than conda)
echo "📦 Installing Python packages in virtual environment..."
pip install datasets>=2.14.0 huggingface_hub>=0.17.0 transformers>=4.30.0
pip install tqdm requests

# Check and mount the EBS data volume
echo "🔍 Checking EBS volumes..."
lsblk

# Wait for EBS volume to be available
sleep 10

# Check if /dev/nvme1n1 (or /dev/xvdf) exists and mount it
DATA_DEVICE=""
if [ -b "/dev/nvme1n1" ]; then
    DATA_DEVICE="/dev/nvme1n1"
elif [ -b "/dev/xvdf" ]; then
    DATA_DEVICE="/dev/xvdf"
else
    echo "❌ No additional EBS volume found! Please check volume attachment."
    lsblk
    exit 1
fi

echo "📦 Found data volume: $DATA_DEVICE"

# Check if volume has a filesystem, create one if not
if ! blkid $DATA_DEVICE; then
    echo "🔧 Formatting EBS volume with ext4..."
    mkfs.ext4 $DATA_DEVICE
fi

# Create mount point and mount the volume
mkdir -p /mnt/imagenet-data
mount $DATA_DEVICE /mnt/imagenet-data

# Verify mount
if ! mountpoint -q /mnt/imagenet-data; then
    echo "❌ Failed to mount EBS volume!"
    exit 1
fi

echo "✅ EBS volume mounted successfully at /mnt/imagenet-data"
df -h /mnt/imagenet-data

# Set permissions
chown ubuntu:ubuntu /mnt/imagenet-data
chmod 755 /mnt/imagenet-data

# Create download script
cat > /home/ubuntu/download_imagenet.sh << 'EODS'
#!/bin/bash
set -e

# Activate the virtual environment
source /opt/imagenet-env/bin/activate

# Check for HuggingFace token (multiple sources)
if [ -n "$1" ]; then
    # Token passed as argument
    HUGGINGFACE_TOKEN="$1"
    export HUGGINGFACE_TOKEN
    echo "🔐 Using HuggingFace token from command line argument"
elif [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "HF_TOKEN_PLACEHOLDER" ]; then
    # Token from environment variable (set during user-data)
    HUGGINGFACE_TOKEN="$HF_TOKEN"
    export HUGGINGFACE_TOKEN
    echo "🔐 Using HuggingFace token from environment variable"
elif [ -n "$HUGGINGFACE_TOKEN" ]; then
    # Token from current environment
    export HUGGINGFACE_TOKEN
    echo "🔐 Using HuggingFace token from current environment"
else
    echo "❌ No HuggingFace token provided!"
    echo "Usage options:"
    echo "  1. ./download_imagenet.sh YOUR_TOKEN_HERE"
    echo "  2. export HUGGINGFACE_TOKEN='your_token' && ./download_imagenet.sh"
    echo "  3. Set HF_TOKEN in user-data script"
    exit 1
fi

echo "✅ Token set: ${HUGGINGFACE_TOKEN:0:10}..."

echo "📥 Starting ImageNet-1k download..."
echo "This will download ~150GB of data. Estimated time: 30-60 minutes on c5n.xlarge"

# Verify mount point has enough space
echo "📊 Available space on data volume:"
df -h /mnt/imagenet-data

# Ensure we're using the mounted volume for all operations
cd /mnt/imagenet-data

# Set environment variables to force all cache to use mounted volume
export HF_DATASETS_CACHE="/mnt/imagenet-data/hf_cache"
export HF_HOME="/mnt/imagenet-data/hf_home"
export TRANSFORMERS_CACHE="/mnt/imagenet-data/transformers_cache"
export TMPDIR="/mnt/imagenet-data/tmp"

# Create necessary directories
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TMPDIR"

echo "🗂️ Cache directories:"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  TMPDIR: $TMPDIR"
echo "  Python environment: $(which python)"

# Login to HuggingFace
python << 'EOF'
import os
from huggingface_hub import login
try:
    login(token=os.environ['HUGGINGFACE_TOKEN'])
    print('✅ HuggingFace login successful')
except Exception as e:
    print(f'❌ HuggingFace login failed: {e}')
    exit(1)
EOF

# Download ImageNet with Python script for better control
python << 'EOF'
import os
import sys
from datasets import load_dataset
from pathlib import Path
import time

try:
    print("📦 Loading ImageNet-1k dataset...")
    print(f"Using cache directory: {os.environ.get('HF_DATASETS_CACHE', 'default')}")
    print(f"Python executable: {sys.executable}")
    
    start_time = time.time()
    
    # Verify we have enough space before starting
    import shutil
    free_space = shutil.disk_usage("/mnt/imagenet-data").free
    free_gb = free_space / (1024**3)
    print(f"📊 Available space: {free_gb:.1f} GB")
    
    if free_gb < 160:  # Need ~150GB + buffer
        print(f"❌ Insufficient space! Need at least 160GB, have {free_gb:.1f}GB")
        sys.exit(1)
    
    # Download train split
    print("📥 Downloading training data...")
    train_dataset = load_dataset(
        "imagenet-1k", 
        split="train",
        cache_dir=os.environ['HF_DATASETS_CACHE'],
        trust_remote_code=False
    )
    
    print("🔄 Checking intermediate space usage...")
    used_space = shutil.disk_usage("/mnt/imagenet-data").used
    used_gb = used_space / (1024**3)
    print(f"📊 Space used after train download: {used_gb:.1f} GB")
    
    # Download validation split  
    print("📥 Downloading validation data...")
    val_dataset = load_dataset(
        "imagenet-1k",
        split="validation", 
        cache_dir=os.environ['HF_DATASETS_CACHE'],
        trust_remote_code=False
    )
    
    end_time = time.time()
    download_time = (end_time - start_time) / 60
    
    # Final space check
    final_used = shutil.disk_usage("/mnt/imagenet-data").used
    final_gb = final_used / (1024**3)
    
    print(f"✅ Download completed in {download_time:.1f} minutes!")
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    print(f"📊 Final space used: {final_gb:.1f} GB")
    
    # Create standard ImageNet directory structure
    os.makedirs("/mnt/imagenet-data/train", exist_ok=True)
    os.makedirs("/mnt/imagenet-data/val", exist_ok=True)
    
    # Create symlinks for easier access
    cache_dir = Path(os.environ['HF_DATASETS_CACHE']) / "imagenet-1k"
    if cache_dir.exists():
        train_link = Path("/mnt/imagenet-data/train_hf")
        val_link = Path("/mnt/imagenet-data/val_hf")
        
        if not train_link.exists():
            train_link.symlink_to(cache_dir)
        if not val_link.exists():
            val_link.symlink_to(cache_dir)
    
    # Create info file
    with open("/mnt/imagenet-data/dataset_info.txt", "w") as f:
        f.write(f"ImageNet-1k Dataset\n")
        f.write(f"Downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training samples: {len(train_dataset)}\n")
        f.write(f"Validation samples: {len(val_dataset)}\n")
        f.write(f"Download time: {download_time:.1f} minutes\n")
        f.write(f"Total size: {final_gb:.1f} GB\n")
        f.write(f"Cache location: {os.environ['HF_DATASETS_CACHE']}\n")
        f.write(f"Python environment: {sys.executable}\n")
        f.write(f"Access via: datasets.load_dataset('imagenet-1k', cache_dir='{os.environ['HF_DATASETS_CACHE']}')\n")
    
    print("📝 Dataset info saved to /mnt/imagenet-data/dataset_info.txt")
    
except Exception as e:
    print(f"❌ Error downloading ImageNet: {e}")
    print("💾 Disk usage at error:")
    os.system("df -h /mnt/imagenet-data")
    os.system("df -h /")
    sys.exit(1)
EOF

echo "✅ ImageNet download completed successfully!"
echo "📊 Dataset size:"
du -sh /mnt/imagenet-data
echo ""
echo "📝 Next steps:"
echo "1. Create EBS snapshot: ./create_snapshot.sh"
echo "2. Use snapshot for training instances"
echo "3. Terminate this download instance to save costs"

EODS

chmod +x /home/ubuntu/download_imagenet.sh
chown ubuntu:ubuntu /home/ubuntu/download_imagenet.sh

# Create snapshot creation script
cat > /home/ubuntu/create_snapshot.sh << 'EOSS'
#!/bin/bash
set -e

# Get the volume ID of the data volume (assuming /dev/nvme1n1 is mounted as /mnt/imagenet-data)
VOLUME_ID=$(aws ec2 describe-volumes \
    --filters "Name=attachment.instance-id,Values=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)" \
    --query 'Volumes[?Attachments[0].Device==`/dev/sdf`].VolumeId' \
    --output text \
    --region us-east-1)

if [ -z "$VOLUME_ID" ]; then
    echo "❌ Could not find data volume. Please check volume attachment."
    exit 1
fi

echo "📸 Creating EBS snapshot of volume $VOLUME_ID..."

SNAPSHOT_ID=$(aws ec2 create-snapshot \
    --volume-id $VOLUME_ID \
    --description "ImageNet-1k Dataset - $(date '+%Y-%m-%d %H:%M:%S')" \
    --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=ImageNet-1k-Dataset},{Key=Project,Value=mosaic-resnet50}]' \
    --query 'SnapshotId' \
    --output text \
    --region us-east-1)

echo "✅ Snapshot created: $SNAPSHOT_ID"
echo "📋 Use this snapshot ID in your training instance launch commands"
echo "💰 You can now safely terminate this download instance"

EOSS

chmod +x /home/ubuntu/create_snapshot.sh
chown ubuntu:ubuntu /home/ubuntu/create_snapshot.sh

# Final setup message
cat > /home/ubuntu/README.txt << 'EORM'
ImageNet Download Instance Setup Complete! 🎉

Virtual Environment: /opt/imagenet-env (automatically activated in scripts)

Usage Options:

1. Pass token as argument:
   ./download_imagenet.sh hf_your_token_here

2. Set environment variable:
   export HUGGINGFACE_TOKEN='hf_your_token_here'
   ./download_imagenet.sh

3. Token was pre-set during launch (if provided in user-data)

Next Steps:
1. SSH into this instance: ssh -i your-key.pem ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

2. Start download (choose one method above)

3. Create EBS snapshot after download:
   ./create_snapshot.sh

4. Terminate this instance to save costs

Environment Details:
- Python virtual environment: /opt/imagenet-env
- Dataset storage: /mnt/imagenet-data (200GB EBS volume)
- Cache directories: All under /mnt/imagenet-data/
- Estimated download time: 30-60 minutes on c5n.xlarge
- Total cost: ~$0.20-0.40 for the entire download process

EORM

echo "🎉 Download instance setup complete!"
echo "📖 See /home/ubuntu/README.txt for next steps"
