#!/bin/bash
echo "🚀 GPU Optimization Helper Script"
echo "=================================="

# Check current GPU usage
echo "📊 Current GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Get memory info
MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
MEMORY_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

echo ""
echo "📈 Optimization Analysis:"
echo "   Memory Usage: ${MEMORY_USED}MB / ${MEMORY_TOTAL}MB"
echo "   GPU Utilization: ${GPU_UTIL}%"

# Calculate memory percentage
MEMORY_PCT=$((MEMORY_USED * 100 / MEMORY_TOTAL))

echo "   Memory Utilization: ${MEMORY_PCT}%"

# Recommend configuration
echo ""
echo "🎯 Recommended Configuration:"

if [ "$MEMORY_PCT" -lt 25 ] && [ "$GPU_UTIL" -lt 70 ]; then
    echo "   🚀 MASSIVE HEADROOM DETECTED!"
    echo "   → Recommended: aws_a10g_max_config (384 batch size, 50K samples)"
    echo ""
    echo "💡 Commands to restart with maximum optimization:"
    echo "   pkill -f python  # Stop current training"
    echo "   ./start_training.sh aws_a10g_max_config"
    echo ""
    echo "⚡ Expected improvements:"
    echo "   - 4-6x higher throughput"
    echo "   - 80-95% GPU utilization"
    echo "   - 15-20GB memory usage"
    echo "   - Same training time, much more data processed"
    
elif [ "$MEMORY_PCT" -lt 50 ] && [ "$GPU_UTIL" -lt 80 ]; then
    echo "   🔧 GOOD OPTIMIZATION OPPORTUNITY"
    echo "   → Recommended: aws_g4dn_validation_config (256 batch size)"
    echo ""
    echo "💡 Commands to restart with optimization:"
    echo "   pkill -f python  # Stop current training"  
    echo "   ./start_training.sh aws_g4dn_validation_config"
    echo ""
    echo "⚡ Expected improvements:"
    echo "   - 2-3x higher throughput"
    echo "   - 70-85% GPU utilization"
    echo "   - Same training quality, faster completion"
    
else
    echo "   ✅ GPU is well utilized!"
    echo "   Current configuration appears optimal for your setup."
fi

echo ""
echo "🔍 Training Process Status:"
if pgrep -f "train.py" > /dev/null; then
    echo "   ✅ Training is currently running"
    echo "   PID: $(pgrep -f train.py)"
else
    echo "   ⚠️  No training process detected"
fi

echo ""
echo "🎮 Available Configurations:"
echo "   1. aws_g4dn_validation_config  → 256 batch size (balanced)"
echo "   2. aws_a10g_max_config         → 384 batch size (maximum)"
echo "   3. aws_g4dn_12xl_ddp_config    → 128 batch size (multi-GPU)"
echo ""
echo "🚀 To restart with new config:"
echo "   pkill -f python && ./start_training.sh [config_name]"
