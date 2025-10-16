#!/bin/bash
# Helper script to launch ImageNet download instance with HuggingFace token
# Usage: ./launch_download_instance.sh YOUR_HF_TOKEN

if [ -z "$1" ]; then
    echo "❌ Usage: $0 YOUR_HF_TOKEN"
    echo "Example: $0 hf_your_token_here"
    exit 1
fi

HF_TOKEN="$1"

# Create temporary user-data script with token embedded
TEMP_SCRIPT=$(mktemp)
sed "s/HF_TOKEN_PLACEHOLDER/$HF_TOKEN/g" scripts/user-data-download.sh > "$TEMP_SCRIPT"

echo "🚀 Launching c5n.xlarge instance for ImageNet download..."
echo "📝 HuggingFace token: ${HF_TOKEN:0:10}..."

# Launch instance with embedded token
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
  --user-data "file://$TEMP_SCRIPT" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=imagenet-download},{Key=Project,Value=mosaic-resnet50}]' \
  --region us-east-1 \
  --profile personal

# Clean up temporary file
rm "$TEMP_SCRIPT"

echo "✅ Instance launched! Check AWS console for status."
echo "📋 To monitor progress:"
echo "   1. Wait 2-3 minutes for instance to boot"
echo "   2. SSH into instance and run: tail -f /var/log/user-data.log"
echo "   3. Download will start automatically with your token"
