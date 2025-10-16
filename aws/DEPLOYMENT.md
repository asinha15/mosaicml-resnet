# AWS Deployment Guide - New Optimized Strategy

## 🎯 3-Phase Strategy (Reliable & Cost-Effective)

### **Phase 1: Fast Dataset Download (c5n.xlarge)**
- **Instance**: c5n.xlarge (network-optimized, on-demand)
- **Purpose**: Download ImageNet-1k quickly and reliably  
- **Cost**: ~$0.30/hr × 1hr = $0.30 total
- **Network**: Up to 25 Gbps for fast downloads
- **Why**: Avoid conda issues, reliable downloads, create reusable EBS snapshot

### **Phase 2: Validation Training (g4dn.xlarge spot)**
- **Instance**: g4dn.xlarge (spot pricing)
- **Purpose**: Quick validation with subset
- **Cost**: ~$0.20/hr × 1hr = $0.20 total
- **Setup**: Pip-based (no conda complexity)

### **Phase 3: Full Training (g4dn.12xlarge spot)**
- **Instance**: g4dn.12xlarge (spot pricing)
- **Purpose**: Full ImageNet training
- **Cost**: ~$1.20/hr × 24hr = $29 total
- **Setup**: Multi-GPU with existing dataset

---

## Phase 1: Dataset Download

> 💡 **Why separate download?** 
> - ✅ **Reliability**: On-demand instances won't get interrupted
> - ✅ **Speed**: c5n instances have 25 Gbps network performance  
> - ✅ **Cost**: Download once, use many times
> - ✅ **No conda issues**: Pure pip-based setup
> - ✅ **Reusable**: Create EBS snapshot for future use

### Prerequisites
> ⚠️ **Important**: ImageNet-1k on HuggingFace requires approval. 
> 1. Request access at: https://huggingface.co/datasets/imagenet-1k
> 2. Create token at: https://huggingface.co/settings/tokens (with read access)
> 3. Save your token - you'll need it for download

### Step 1: Launch Download Instance

**Recommended Instance: c5n.xlarge**
- **vCPUs**: 4
- **Memory**: 10.5 GB  
- **Network**: Up to 25 Gbps
- **Cost**: ~$0.216/hour

```bash
# Launch c5n.xlarge for fast dataset download (on-demand for reliability)
aws ec2 run-instances \
  --image-id ami-0ac1f653c5b6af751 \
  --instance-type c5n.xlarge \
  --key-name aws-kp-lvboy \
  --security-group-ids sg-1c81e178 \
  --subnet-id subnet-4e67cc65 \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {"VolumeSize": 50, "VolumeType": "gp3"}
  }, {
    "DeviceName": "/dev/sdf", 
    "Ebs": {"VolumeSize": 200, "VolumeType": "gp3"}
  }]' \
  --user-data file://scripts/user-data-download.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=imagenet-download},{Key=Project,Value=mosaic-resnet50}]' \
  --region us-east-1 \
  --profile personal
```

> 📝 **What user-data-download.sh does:**
> - Installs Python 3 + pip (no conda complexity)
> - Sets up HuggingFace datasets library
> - Creates optimized download scripts  
> - Prepares EBS snapshot creation tools
> - Mounts 200GB EBS volume for dataset storage

### Step 2: Download Dataset

```bash
# SSH into download instance
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=imagenet-download" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text --region us-east-1 --profile personal)

ssh -i your-key.pem ubuntu@$INSTANCE_IP

# Set your HuggingFace token
export HUGGINGFACE_TOKEN='your_token_here'

# Start download (estimated 30-60 minutes on c5n.xlarge)
./download_imagenet.sh

# Monitor progress
tail -f /var/log/user-data.log
```

### Step 3: Create EBS Snapshot

```bash
# After download completes, create snapshot for reuse
./create_snapshot.sh

# Note the snapshot ID for Phase 2/3
# Example output: snap-0123456789abcdef0
```

### Step 4: Terminate Download Instance
```bash
# After snapshot creation, terminate to save costs
aws ec2 terminate-instances \
  --instance-ids $(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=imagenet-download" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text --region us-east-1 --profile personal) \
  --region us-east-1 --profile personal
```

---

## Phase 2: Validation Training

### Step 1: Launch Training Instance

```bash
# Create EBS volume from Phase 1 snapshot
SNAPSHOT_ID="snap-08af26be1b5312b42"  # From Phase 1
VOLUME_ID=$(aws ec2 create-volume \
  --snapshot-id $SNAPSHOT_ID \
  --availability-zone us-east-1a \
  --size 200 \
  --volume-type gp3 \
  --region us-east-1 \
  --profile personal \
  --query "VolumeId" --output text)

# Launch g4dn.xlarge SPOT instance for validation
aws ec2 run-instances \
  --image-id ami-0ac1f653c5b6af751 \
  --instance-type g4dn.xlarge \
  --key-name aws-kp-lvboy \
  --security-group-ids sg-1c81e178 \
  --subnet-id subnet-4e67cc65 \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "MaxPrice": "0.50",
      "SpotInstanceType": "one-time",
      "InstanceInterruptionBehavior": "terminate"
    }
  }' \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
  }]' \
  --user-data file://scripts/user-data-training.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=resnet50-validation},{Key=Project,Value=mosaic-resnet50}]' \
  --region us-east-1 \
  --profile personal
```

### Step 2: Attach ImageNet Volume

```bash
# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=resnet50-validation" "Name=instance-state-name,Values=running" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text --region us-east-1 --profile personal)

# Attach ImageNet volume
aws ec2 attach-volume \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf \
  --region us-east-1 \
  --profile personal
```

### Step 3: Start Validation Training

```bash
# SSH into training instance
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=resnet50-validation" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text --region us-east-1 --profile personal)

ssh -i your-key.pem ubuntu@$INSTANCE_IP

# Mount ImageNet volume
sudo mkdir -p /mnt/imagenet-data
sudo mount /dev/nvme1n1 /mnt/imagenet-data

# Activate training environment
source activate_env.sh

# Start validation training (subset)
./start_training.sh
```

---

## Phase 3: Full Production Training

### Step 1: Launch Production Instance

```bash
# Launch g4dn.12xlarge SPOT instance for full training
aws ec2 run-instances \
  --image-id ami-0ac1f653c5b6af751 \
  --instance-type g4dn.12xlarge \
  --key-name aws-kp-lvboy \
  --security-group-ids sg-1c81e178 \
  --subnet-id subnet-4e67cc65 \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "MaxPrice": "2.00",
      "SpotInstanceType": "one-time",
      "InstanceInterruptionBehavior": "terminate"
    }
  }' \
  --user-data file://scripts/user-data-training.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=resnet50-production},{Key=Project,Value=mosaic-resnet50}]' \
  --region us-east-1 \
  --profile personal

# Attach existing ImageNet volume (reuse from Phase 2 or create new from snapshot)
```

---

## 💡 Key Benefits of New Strategy

### **Reliability**
- ✅ No conda ToS issues (pure pip)
- ✅ No package installation failures causing shutdowns
- ✅ Separate download from training (no interruption risk)
- ✅ Reusable EBS snapshots

### **Performance**  
- ✅ c5n.xlarge: 25 Gbps network for fast downloads
- ✅ Optimized for network-intensive operations
- ✅ No memory issues during package installation

### **Cost Optimization**
- ✅ Download: $0.30 total (1 hour c5n.xlarge)
- ✅ Validation: $0.20 total (1 hour g4dn.xlarge spot)
- ✅ Full training: ~$29 total (24 hours g4dn.12xlarge spot)
- ✅ **Total project cost: ~$30** vs $120+ with on-demand

### **Simplicity**
- ✅ Pure pip installation (no conda complexity)
- ✅ Python venv (lightweight, reliable)
- ✅ Clear separation of concerns
- ✅ Easy to debug and reproduce

---

## 🚀 Quick Start Commands

```bash
# 1. Download dataset (Phase 1)
aws ec2 run-instances --instance-type c5n.xlarge --user-data file://scripts/user-data-download.sh ...

# 2. Validation training (Phase 2)  
aws ec2 run-instances --instance-type g4dn.xlarge --user-data file://scripts/user-data-training.sh ...

# 3. Full training (Phase 3)
aws ec2 run-instances --instance-type g4dn.12xlarge --user-data file://scripts/user-data-training.sh ...
```

## 📊 Instance Comparison

| Instance | Type | vCPU | Memory | GPU | Network | Use Case | Cost/hr |
|----------|------|------|--------|-----|---------|----------|---------|
| c5n.xlarge | Compute | 4 | 10.5GB | None | 25 Gbps | Download | $0.216 |
| g4dn.xlarge | GPU | 4 | 16GB | 1×T4 | 25 Gbps | Validation | $0.20 spot |
| g4dn.12xlarge | GPU | 48 | 192GB | 4×T4 | 50 Gbps | Production | $1.20 spot |

🎯 **Perfect balance of speed, reliability, and cost!**
