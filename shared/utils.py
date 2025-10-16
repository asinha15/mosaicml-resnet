"""
Utility functions for ResNet50 training project.
"""
import os
import json
import yaml
import torch
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str, config_name: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if config_name not in configs:
        raise ValueError(f"Configuration '{config_name}' not found in {config_path}")
    
    return configs[config_name]


def setup_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create experiment directory structure."""
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    (exp_dir / 'configs').mkdir(exist_ok=True)
    
    return exp_dir


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_system_info() -> Dict[str, Any]:
    """Get system information for logging."""
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    
    return info


def log_system_info():
    """Log system information to console and wandb if available."""
    info = get_system_info()
    
    print("🖥️  System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Log to wandb if active
    if wandb.run is not None:
        wandb.log({'system_info': info})


def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def log_model_info(model: torch.nn.Module, model_name: str = "ResNet50"):
    """Log model information."""
    stats = calculate_model_size(model)
    
    print(f"📊 {model_name} Model Statistics:")
    print(f"  Total parameters: {stats['total_parameters']:,}")
    print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
    print(f"  Non-trainable parameters: {stats['non_trainable_parameters']:,}")
    
    # Calculate model size in MB
    param_size_mb = stats['total_parameters'] * 4 / (1024 ** 2)  # Assuming float32
    print(f"  Model size: {param_size_mb:.1f} MB")
    
    # Log to wandb if active
    if wandb.run is not None:
        wandb.log({'model_stats': stats})


def plot_training_curves(metrics_file: str, save_dir: str):
    """Plot training curves from saved metrics."""
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Metrics file {metrics_file} not found")
        return
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Training and validation loss
    if 'train_loss' in metrics and 'val_loss' in metrics:
        axes[0, 0].plot(metrics['train_loss'], label='Train Loss')
        axes[0, 0].plot(metrics['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Accuracy curves
    if 'train_acc' in metrics and 'val_acc' in metrics:
        axes[0, 1].plot(metrics['train_acc'], label='Train Acc')
        axes[0, 1].plot(metrics['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Learning rate schedule
    if 'learning_rate' in metrics:
        axes[1, 0].plot(metrics['learning_rate'])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
    
    # GPU memory usage
    if 'gpu_memory' in metrics:
        axes[1, 1].plot(metrics['gpu_memory'])
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Memory (GB)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    
    # Log to wandb if active
    if wandb.run is not None:
        wandb.log({"training_curves": wandb.Image(save_path)})


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    epochs: int,
    samples_per_second: float
) -> Dict[str, float]:
    """Estimate training time based on throughput."""
    total_samples = num_samples * epochs
    total_seconds = total_samples / samples_per_second
    
    return {
        'total_hours': total_seconds / 3600,
        'total_minutes': total_seconds / 60,
        'hours_per_epoch': total_seconds / epochs / 3600,
        'samples_per_second': samples_per_second
    }


def check_gpu_memory():
    """Check and log GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"🔧 GPU Memory Status:")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated
    }


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 5):
    """Clean up old checkpoints, keeping only the most recent."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_path.glob("*.pt"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    
    # Remove old checkpoints
    if len(checkpoints) > keep_last_n:
        old_checkpoints = checkpoints[:-keep_last_n]
        for checkpoint in old_checkpoints:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint}")


def create_wandb_config(args) -> Dict[str, Any]:
    """Create wandb config from training arguments."""
    config = {}
    for key, value in vars(args).items():
        if not key.startswith('_'):
            config[key] = value
    
    # Add system info
    config.update(get_system_info())
    
    return config


class TrainingLogger:
    """Simple training logger that works with or without wandb."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.metrics_history = {}
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to console, file, and wandb if available."""
        # Console logging
        if step is not None:
            print(f"Step {step}: ", end="")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}={value:.4f} ", end="")
        print()  # New line
        
        # File logging
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Step {step}: {metrics}\n")
        
        # History tracking
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # wandb logging
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    
    def save_metrics(self, save_path: str):
        """Save metrics history to JSON file."""
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
