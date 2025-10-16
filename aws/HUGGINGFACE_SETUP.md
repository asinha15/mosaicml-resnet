# HuggingFace ImageNet-1k Access Guide

## 🔐 Getting ImageNet Access

ImageNet-1k on HuggingFace requires approval and authentication. Follow these steps:

### Step 1: Request Dataset Access

1. **Visit the dataset page**: https://huggingface.co/datasets/imagenet-1k
2. **Click "Request Access"** - you'll need to agree to terms
3. **Wait for approval** - usually takes 1-2 business days
4. **You'll receive an email** when approved

### Step 2: Create HuggingFace Token

1. **Login to HuggingFace**: https://huggingface.co/login  
2. **Go to Settings**: https://huggingface.co/settings/tokens
3. **Create New Token**:
   - Name: `imagenet-training` (or similar)
   - Type: `Read` (sufficient for downloading datasets)
   - Click "Create"
4. **Copy the token** - starts with `hf_...`

> ⚠️ **Important**: Save your token securely! You won't be able to see it again.

### Step 3: Use Token with Setup Script

**Option 1: Environment Variable (Recommended)**
```bash
export HUGGINGFACE_TOKEN="hf_your_actual_token_here"
./scripts/setup_imagenet.sh setup
```

**Option 2: Interactive Input**
```bash
./scripts/setup_imagenet.sh setup
# Script will prompt: "Enter your HuggingFace token: "
# Paste your token and optionally save it for future use
```

**Option 3: Pre-save Token**
```bash
# Save token to HF cache (secure, 600 permissions)
mkdir -p ~/.cache/huggingface
echo "hf_your_actual_token_here" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

# Now script will auto-detect the token
./scripts/setup_imagenet.sh setup
```

## 🚨 Troubleshooting

### "Access Denied" Errors

**Error**: `HTTPError: 403 Client Error: Forbidden`

**Solutions**:
1. **Check dataset access**: Ensure you've been approved for imagenet-1k dataset
2. **Verify token**: Make sure token is valid and has read permissions
3. **Re-authenticate**: Try `huggingface-cli login --token your_token`

### "Invalid Token" Errors

**Error**: `Invalid token` or authentication failures

**Solutions**:
1. **Check token format**: Should start with `hf_`
2. **Recreate token**: Create a new token at https://huggingface.co/settings/tokens
3. **Check permissions**: Ensure token has at least `Read` access

### "Dataset Not Found" Errors

**Error**: `DatasetNotFoundError: Dataset 'imagenet-1k' doesn't exist`

**Solutions**:
1. **Wait for approval**: Dataset won't be visible until access is approved
2. **Check approval status**: Visit https://huggingface.co/datasets/imagenet-1k
3. **Contact support**: If approved but still can't access, contact HuggingFace support

## 📊 What Gets Downloaded

When successful, the script downloads:

- **Train split**: ~1.28M images (~138GB)
- **Validation split**: ~50K images (~6GB) 
- **Total size**: ~150GB (compressed)
- **Cache location**: `/opt/imagenet/hf_cache/`
- **Download time**: 30-60 minutes (depends on connection)

## 🔒 Security Best Practices

1. **Never share tokens**: Keep your HF token private
2. **Use environment variables**: Don't hardcode tokens in scripts
3. **Secure permissions**: Token files should be `chmod 600`
4. **Rotate tokens**: Periodically create new tokens and delete old ones
5. **Minimum permissions**: Use `Read` access only for training purposes

## 💡 Alternative: Manual Download

If HuggingFace access is problematic, you can:

1. **Download ImageNet manually** from http://www.image-net.org/
2. **Convert to HuggingFace format** using their conversion tools
3. **Use local ImageNet** by setting `use_hf: false` in training config

However, the HuggingFace approach is **strongly recommended** for ease of use and consistency.

---

**Next**: Once you have ImageNet downloaded, proceed with [Phase 2 training](DEPLOYMENT.md#phase-2-aws-single-gpu-validation)
