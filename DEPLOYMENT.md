# Multi-Phase ResNet50 Deployment Guide

Complete deployment guide for the 3-phase ResNet50 ImageNet training on AWS.

## 🎯 Phase Overview

| Phase | Objective | Hardware | Time | Cost | Purpose |
|-------|-----------|----------|------|------|---------|
| **Phase 1** | ✅ Pipeline validation | Colab T4 | 10 min | Free | Ensure everything works |
| **Phase 2** | Infrastructure validation | g4dn.xlarge | 1 hour | ~$0.50 | Validate AWS setup |
| **Phase 3** | Production training | g4dn.12xlarge | 16 hours | ~$65 | Achieve >78% accuracy |

## 📊 Phase 2: AWS Single GPU Validation

### Prerequisites

- AWS account with EC2 access
- **HuggingFace account with ImageNet-1k access** (required!)
- **HuggingFace token** with read access
- Basic familiarity with AWS console
- SSH key pair configured

> ⚠️ **Important**: ImageNet-1k on HuggingFace requires approval. 
> 1. Request access at: https://huggingface.co/datasets/imagenet-1k
> 2. Create token at: https://huggingface.co/settings/tokens (with read access)
> 3. Save your token - you'll need it for Phase 2 setup

### Step 1: Launch Instance

```bash
# Launch g4dn.xlarge instance with user-data for automatic setup
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \  # Deep Learning AMI (Ubuntu)
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-your-security-group \
  --subnet-id subnet-your-subnet \
  --block-device-mappings '[{
    "DeviceName": "/dev/sda1",
    "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}
  }, {
    "DeviceName": "/dev/sdf", 
    "Ebs": {"VolumeSize": 200, "VolumeType": "gp3"}
  }]' \
  --user-data file://scripts/user-data-phase2.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=resnet50-phase2},{Key=Project,Value=mosaic-resnet50}]'
```

> 📝 **What user-data-phase2.sh does:**
> - Installs system dependencies (git, htop, nvtop)  
> - Sets up Miniconda with Python 3.10
> - Installs PyTorch + MosaicML Composer
> - Creates project directories
> - Prepares environment for ImageNet download

### Step 2: Instance Setup

SSH into your instance:

```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Run setup script (automatically detects g4dn.xlarge)
curl -sL https://raw.githubusercontent.com/yourrepo/mosaic-resnet/main/scripts/aws_setup.sh | bash

# Setup ImageNet dataset with your HuggingFace token
export HUGGINGFACE_TOKEN="hf_your_token_here"  # Replace with your actual token
./scripts/setup_imagenet.sh setup

# Alternative: Script will prompt for token if not set
# ./scripts/setup_imagenet.sh setup
```

> 💡 **Token Security**: The script can save your token securely for reuse, or you can set it as an environment variable each time.

### Step 3: Run Phase 2 Training

```bash
# Activate environment
conda activate resnet50

# Configure W&B (optional but recommended)
wandb login

# Run 1-hour validation training
./train_phase2.sh
```

### Step 4: Validation Results

Expected output after ~1 hour:
```
Phase 2 Validation Results:
- Dataset: 25,000 ImageNet samples
- Final accuracy: ~40-50% (normal for subset)
- Training speed: ~800-1000 samples/sec
- GPU memory: ~12-14GB used
- Total time: ~55-65 minutes
```

### Step 5: Create EBS Snapshot

```bash
# Create snapshot for Phase 3 reuse
./scripts/setup_imagenet.sh snapshot

# Output will be snapshot ID like: snap-abc123def456
# SAVE THIS ID for Phase 3!
```

## 🚀 Phase 3: AWS Multi-GPU Production

### Prerequisites

- Successful Phase 2 completion
- EBS snapshot ID from Phase 2
- Budget approval for ~$65 training run

### Step 1: Launch Production Instance

```bash
# Create EBS volume from Phase 2 snapshot
SNAPSHOT_ID="snap-abc123def456"  # From Phase 2
VOLUME_ID=$(aws ec2 create-volume \
  --snapshot-id $SNAPSHOT_ID \
  --availability-zone us-west-2a \
  --size 200 \
  --volume-type gp3 \
  --query "VolumeId" --output text)

# Launch g4dn.12xlarge with existing ImageNet volume
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.12xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-your-security-group \
  --subnet-id subnet-your-subnet \
  --user-data file://user-data-phase3.sh

# Attach ImageNet volume
aws ec2 attach-volume \
  --volume-id $VOLUME_ID \
  --instance-id $INSTANCE_ID \
  --device /dev/sdf
```

### Step 2: Production Setup

```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Run setup (detects g4dn.12xlarge automatically)
curl -sL https://raw.githubusercontent.com/yourrepo/mosaic-resnet/main/scripts/aws_setup.sh | bash

# Mount existing ImageNet data
./scripts/setup_imagenet.sh ebs-only

# Verify 4 GPUs detected
nvidia-smi
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Step 3: Run Production Training

```bash
# Activate environment  
conda activate resnet50

# Configure W&B with production project
wandb login

# Start full ImageNet training with DDP
./train_phase3.sh

# Monitor progress (in separate terminal)
tail -f /opt/checkpoints/train.log
watch -n 10 nvidia-smi
```

### Step 4: Monitor Training Progress

Key metrics to watch:

```bash
# Training progress (should show 4 GPU utilization)
nvidia-smi

# W&B dashboard metrics:
- Training loss: Should decrease steadily
- Validation accuracy: Target >78% by epoch 90
- GPU memory: ~14GB per GPU
- Training speed: ~2000+ samples/sec total

# Estimated timeline:
- Hours 0-2: Initial epochs, accuracy ~5-15%
- Hours 4-8: Mid training, accuracy ~40-60% 
- Hours 12-16: Final epochs, accuracy >75%
```

### Step 5: Production Results

Expected final results:
```
Phase 3 Production Results:
- Dataset: 1,281,167 ImageNet training samples
- Final top-1 accuracy: >78% (target achieved)
- Final top-5 accuracy: >94% (typical)
- Total training time: 12-16 hours
- Peak throughput: ~2000+ samples/sec
- Total cost: ~$60-80
```

## 💰 Cost Optimization

### Phase 2 Cost Breakdown
- g4dn.xlarge: $0.526/hour × 1 hour = ~$0.53
- EBS storage: $0.10/GB × 200GB × 1 day = ~$0.66
- **Total Phase 2: ~$1.20**

### Phase 3 Cost Breakdown  
- g4dn.12xlarge: $4.277/hour × 16 hours = ~$68
- EBS storage: Already created in Phase 2
- **Total Phase 3: ~$68**

### Cost Optimization Tips

1. **Use Spot Instances**: 50-70% savings
   ```bash
   aws ec2 request-spot-instances --spot-price "2.50" --instance-count 1 --type "one-time" --launch-specification file://spot-spec.json
   ```

2. **Automatic Termination**: Set CloudWatch alarm
   ```bash
   # Auto-terminate after training completes
   aws cloudwatch put-metric-alarm --alarm-name "terminate-after-training" --alarm-actions "arn:aws:automate:region:account:ec2:terminate"
   ```

3. **EBS Snapshot Cleanup**: Delete old snapshots
   ```bash
   # Keep only latest snapshot
   aws ec2 describe-snapshots --owner-ids self --query 'Snapshots[?Description==`ImageNet-1K*`]'
   ```

## 🔍 Troubleshooting

### Common Phase 2 Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| ImageNet download fails | `trust_remote_code` error | Use updated scripts (fixed) |
| GPU not detected | CPU training | Check Deep Learning AMI |
| OOM errors | Training crashes | Reduce batch size to 128 |

### Common Phase 3 Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Only 1 GPU used | Low throughput | Verify DDP config |
| Network bottleneck | Slow data loading | Use local EBS, not S3 |
| Training diverges | Loss increases | Lower learning rate |

### Debug Commands

```bash
# Check GPU utilization
nvidia-smi -l 1

# Check training logs
tail -f /opt/checkpoints/$(ls -t /opt/checkpoints/ | head -1)/train.log

# Check disk I/O
iostat -x 1

# Check network usage (should be minimal with local data)
iftop -i eth0
```

## ✅ Success Criteria

### Phase 2 Success
- [ ] Training completes in ~1 hour
- [ ] No CUDA/GPU errors
- [ ] Achieves ~40-50% accuracy on subset
- [ ] EBS snapshot created successfully

### Phase 3 Success  
- [ ] All 4 GPUs utilized (nvidia-smi shows >80% utilization)
- [ ] Training speed >1500 samples/sec total
- [ ] **Final top-1 accuracy >78%**
- [ ] No memory or stability issues

## 📋 Next Steps After Success

1. **Model Deployment**: Export to ONNX/TorchScript for inference
2. **Hyperparameter Tuning**: Try different optimizers, schedules
3. **Architecture Experiments**: Test other models (EfficientNet, Vision Transformer)
4. **Production Pipeline**: Set up automated training/evaluation

---

🎉 **Congratulations!** You've successfully trained a >78% accuracy ResNet50 on ImageNet using MosaicML Composer with multi-GPU DDP on AWS!
