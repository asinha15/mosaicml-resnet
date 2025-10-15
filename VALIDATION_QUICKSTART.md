# 🚀 Quick Start: 1-Hour Validation Test

## 📋 **Prerequisites**
1. ✅ ImageNet dataset downloaded (using `user-data-download.sh`)
2. ✅ EBS snapshot created from download instance
3. ✅ AWS credentials configured

## ⚡ **Launch Validation Instance**

### **Step 1: Create Volume from Snapshot**
```bash
# Create EBS volume from your ImageNet snapshot
SNAPSHOT_ID="snap-08af26be1b5312b42"
VOLUME_ID=$(aws ec2 create-volume \
  --snapshot-id $SNAPSHOT_ID \
  --availability-zone us-east-1a \
  --size 200 \
  --volume-type gp3 \
  --region us-east-1 \
  --profile personal \
  --query "VolumeId" --output text)

echo "Volume created: $VOLUME_ID"
```

### **Step 2: Launch g4dn.xlarge Training Instance**
```bash
# Launch g4dn.xlarge SPOT instance
aws ec2 run-instances \
  --image-id ami-0ac1f653c5b6af751 \
  --instance-type g4dn.4xlarge \
  --key-name aws-kp-lvboy \
  --security-group-ids sg-1c81e178 \
  --subnet-id subnet-006eed59 \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "MaxPrice": "0.99",
      "SpotInstanceType": "one-time",
      "InstanceInterruptionBehavior": "terminate"
    }
  }' \
  --user-data file://scripts/user-data-training.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=resnet50-validation},{Key=Project,Value=mosaic-resnet50}]' \
  --region us-east-1 \
  --profile personal
```

### **Step 3: Attach ImageNet Volume**
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

## 🎯 **Start 1-Hour Validation Test**

### **Step 4: SSH and Start Training**
```bash
# Get instance IP
INSTANCE_IP=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=resnet50-validation" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text --region us-east-1 --profile personal)

# SSH into instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP

# Mount ImageNet volume
sudo mkdir -p /mnt/imagenet-data
sudo mount /dev/nvme1n1 /mnt/imagenet-data

# Start 1-hour validation test
./start_validation.sh
```

## 📊 **What the Validation Test Does**

### **Configuration: `aws_g4dn_validation_config`**
- 📦 **Dataset**: 25,000 samples (2% of ImageNet)
- ⏱️ **Duration**: ~1 hour
- 🎯 **Epochs**: 10
- 🔥 **Batch Size**: 256 (optimized for single T4)
- 🧠 **GPU**: Single T4 (16GB VRAM)
- 💾 **Memory**: Efficient for g4dn.xlarge

### **Composer Optimizations Enabled:**
- ✅ MixUp augmentation
- ✅ CutMix augmentation  
- ✅ RandAugment
- ✅ Label smoothing
- ✅ Exponential Moving Average (EMA)
- ✅ Channels Last memory format
- ✅ BlurPool anti-aliasing

## 📈 **Monitor Progress**

```bash
# Check GPU utilization
nvidia-smi

# Monitor training logs
tail -f /opt/logs/training_aws_g4dn_validation_config_*.log

# Check training progress
check_logs
```

## 🎉 **Expected Results**

After 1 hour, you should see:
- ✅ **Training completes** without errors
- ✅ **GPU utilization** >90%
- ✅ **Memory usage** stable
- ✅ **Checkpoints saved** every 2 epochs
- ✅ **Validation accuracy** improving

## 💰 **Cost Estimate**
- **Instance**: g4dn.xlarge spot (~$0.20/hr × 1hr = $0.20)
- **Storage**: 200GB EBS (~$0.02/hr × 1hr = $0.02)
- **Total**: ~$0.22 for complete validation test

## 🚀 **Next Steps After Validation**
1. ✅ Verify training pipeline works end-to-end
2. ✅ Check performance metrics and GPU utilization
3. ✅ Scale up to full training on g4dn.12xlarge (4x T4)
4. ✅ Use `aws_g4dn_12xl_ddp_config` for production training

Perfect for validating your setup before expensive full training! 🎯
