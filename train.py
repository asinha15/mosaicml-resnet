"""
Main training script for ResNet50 on ImageNet using MosaicML Composer.
Includes comprehensive optimizations and configurations for different scenarios.
"""
import os
import argparse
from pathlib import Path
import torch
import wandb
from composer import Trainer
from composer.algorithms import (
    BlurPool, ChannelsLast, EMA, LabelSmoothing,
    MixUp, CutMix, RandAugment, TrivialAugment,
    LayerFreezing, SAM, SWA
)
from composer.callbacks import (
    LRMonitor, MemoryMonitor, SpeedMonitor,
    ThresholdStopper, EarlyStopper
)
from composer.loggers import WandBLogger
from composer.optim import DecoupledSGDW, DecoupledAdamW
from composer.optim.scheduler import CosineAnnealingWithWarmupScheduler
from composer.utils import dist, reproducibility

from model import create_resnet50_composer
from data_utils import create_dataloaders, get_subset_size


def setup_args() -> argparse.Namespace:
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet with Composer')
    
    # Model arguments
    parser.add_argument('--model-type', default='torchvision', choices=['torchvision', 'custom'],
                      help='Model implementation to use')
    parser.add_argument('--pretrained', action='store_true', 
                      help='Use pretrained weights')
    parser.add_argument('--compile-model', action='store_true',
                      help='Use torch.compile for optimization')
    
    # Data arguments  
    parser.add_argument('--data-subset', default='full', 
                      help='Dataset subset size (tiny/small/medium/large/full or integer)')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='Batch size for training')
    parser.add_argument('--image-size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--num-workers', type=int, default=8,
                      help='Number of data loading workers')
    parser.add_argument('--use-hf', action='store_true', default=True,
                      help='Use HuggingFace dataset instead of local ImageNet')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=90,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='SGD momentum')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adamw'],
                      help='Optimizer to use')
    
    # Composer algorithm arguments
    parser.add_argument('--use-mixup', action='store_true', default=True,
                      help='Enable MixUp augmentation')
    parser.add_argument('--use-cutmix', action='store_true', default=True, 
                      help='Enable CutMix augmentation')
    parser.add_argument('--use-randaugment', action='store_true', default=True,
                      help='Enable RandAugment')
    parser.add_argument('--use-label-smoothing', action='store_true', default=True,
                      help='Enable label smoothing')
    parser.add_argument('--use-ema', action='store_true', default=True,
                      help='Enable Exponential Moving Average')
    parser.add_argument('--use-channels-last', action='store_true', default=True,
                      help='Enable channels last memory format')
    parser.add_argument('--use-blurpool', action='store_true', default=True,
                      help='Enable BlurPool anti-aliasing')
    parser.add_argument('--use-sam', action='store_true', 
                      help='Enable Sharpness Aware Minimization')
    parser.add_argument('--use-swa', action='store_true',
                      help='Enable Stochastic Weight Averaging')
    
    # Infrastructure arguments
    parser.add_argument('--device', default='auto', 
                      help='Device to train on (auto/cpu/gpu/mps)')
    parser.add_argument('--precision', default='amp_fp16', 
                      choices=['fp32', 'amp_fp16', 'amp_bf16'],
                      help='Training precision')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                      help='Gradient clipping norm')
    parser.add_argument('--save-folder', default='./checkpoints',
                      help='Folder to save checkpoints')
    parser.add_argument('--save-interval', default='1ep',
                      help='Checkpoint saving interval')
    
    # Logging arguments
    parser.add_argument('--wandb-project', default='mosaic-resnet50',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', help='Weights & Biases entity')
    parser.add_argument('--experiment-name', help='Experiment name for logging')
    parser.add_argument('--log-interval', default='1ba',
                      help='Logging interval')
    
    # Testing arguments
    parser.add_argument('--dry-run', action='store_true',
                      help='Run a quick test without full training')
    parser.add_argument('--find-lr', action='store_true',
                      help='Run learning rate finder')
    
    return parser.parse_args()


def setup_composer_algorithms(args) -> list:
    """Setup Composer optimization algorithms based on arguments."""
    algorithms = []
    
    if args.use_mixup:
        algorithms.append(MixUp(alpha=0.2))
    
    if args.use_cutmix:
        algorithms.append(CutMix(alpha=1.0))
    
    if args.use_randaugment:
        algorithms.append(RandAugment(severity=9, depth=2))
    
    if args.use_label_smoothing:
        algorithms.append(LabelSmoothing(smoothing=0.1))
    
    if args.use_ema:
        algorithms.append(EMA(half_life='50ba', update_interval='1ba'))
    
    if args.use_channels_last:
        algorithms.append(ChannelsLast())
    
    if args.use_blurpool:
        algorithms.append(BlurPool(
            replace_convs=True,
            replace_maxpools=True,
            blur_first=True
        ))
    
    if args.use_sam:
        algorithms.append(SAM(rho=0.05, adaptive=True))
    
    if args.use_swa:
        algorithms.append(SWA(
            swa_start='10ep',
            swa_end=None,
            update_interval='1ep',
            schedule_swa_lr=True
        ))
    
    return algorithms


def setup_callbacks(args) -> list:
    """Setup training callbacks."""
    callbacks = [
        LRMonitor(),
        MemoryMonitor(),
        SpeedMonitor(window_size=100),
    ]
    
    # Early stopping for validation accuracy
    if not args.dry_run:
        callbacks.append(EarlyStopper(
            monitor='MulticlassAccuracy',
            dataloader_label='val',
            patience='5ep',
            comp=torch.greater,
            min_delta=0.001
        ))
    
    return callbacks


def setup_loggers(args) -> list:
    """Setup logging."""
    loggers = []
    
    if not args.dry_run and args.wandb_project:
        loggers.append(WandBLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            tags=['resnet50', 'imagenet', 'composer']
        ))
    
    return loggers


def create_optimizer(model, args):
    """Create optimizer based on arguments."""
    if args.optimizer == 'sgd':
        return DecoupledSGDW(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        return DecoupledAdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def create_scheduler(optimizer, args, train_dataloader):
    """Create learning rate scheduler with proper time units."""
    # For very short training (< 5 epochs), use batch-based warmup
    if args.epochs < 5:
        warmup_batches = max(1, len(train_dataloader) // 4)  # 25% of first epoch
        return CosineAnnealingWithWarmupScheduler(
            t_warmup=f'{warmup_batches}ba',  # Use batch units for short training
            t_max=f'{args.epochs}ep'
        )
    else:
        return CosineAnnealingWithWarmupScheduler(
            t_warmup='5ep',  # Use epoch units for longer training
            t_max=f'{args.epochs}ep'
        )


def run_lr_finder(model, train_dataloader, args):
    """Run learning rate finder to find optimal learning rate."""
    from composer.optim.scheduler import LinearScheduler
    
    print("Running Learning Rate Finder...")
    
    # Create a temporary trainer for LR finding
    lr_finder_optimizer = create_optimizer(model, args)
    lr_scheduler = LinearScheduler(
        alpha_i=1e-7,
        alpha_f=10.0,
        t_max='100ba'
    )
    
    lr_finder_trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        optimizers=lr_finder_optimizer,
        schedulers=lr_scheduler,
        max_duration='100ba',
        device=args.device,
        precision=args.precision,
        algorithms=[],  # No algorithms for LR finding
        loggers=[],
        callbacks=[LRMonitor(), MemoryMonitor()]
    )
    
    lr_finder_trainer.fit()
    
    print("Learning Rate Finder completed. Check logs for optimal LR.")
    print("Recommended: Use LR around where loss decreases fastest.")


def main():
    """Main training function."""
    args = setup_args()
    
    # Set up reproducibility
    reproducibility.seed_all(42)
    
    # Determine device (Composer uses 'gpu' instead of 'cuda')
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'gpu'  # Composer expects 'gpu' not 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps' 
        else:
            device = 'cpu'
        args.device = device
    
    print(f"Using device: {args.device}")
    print(f"Training precision: {args.precision}")
    
    # Create model
    print("Creating model...")
    model = create_resnet50_composer(
        num_classes=1000,
        pretrained=args.pretrained,
        compile_model=args.compile_model
    )
    
    # Handle device placement: Composer uses 'gpu' but PyTorch uses 'cuda'
    if args.device == 'gpu':
        pytorch_device = 'cuda'
        print(f"Moving model to CUDA (Composer device: {args.device})")
        # Note: Composer Trainer will handle device placement, but we can pre-place for consistency
    else:
        pytorch_device = args.device
        print(f"Using device: {pytorch_device}")
    
    # Create dataloaders
    print("Setting up data...")
    subset_size = get_subset_size(args.data_subset)
    if args.dry_run:
        subset_size = 1000  # Force small subset for dry run
    
    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=subset_size,
        image_size=args.image_size,
        use_hf=args.use_hf
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, train_dataloader)
    
    # Run learning rate finder if requested
    if args.find_lr:
        run_lr_finder(model, train_dataloader, args)
        return
    
    # Setup Composer components
    algorithms = setup_composer_algorithms(args)
    callbacks = setup_callbacks(args)
    loggers = setup_loggers(args)
    
    print(f"Using {len(algorithms)} Composer algorithms:")
    for alg in algorithms:
        print(f"  - {alg.__class__.__name__}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=f'{args.epochs}ep' if not args.dry_run else '2ba',
        eval_interval='1ep' if not args.dry_run else '1ba',
        device=args.device,
        precision=args.precision,
        algorithms=algorithms,
        callbacks=callbacks,
        loggers=loggers,
        save_folder=args.save_folder,
        save_interval=args.save_interval if not args.dry_run else None,
        grad_clip_norm=args.grad_clip_norm,
        seed=42
    )
    
    # Start training
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer.fit()
    
    print("Training completed!")
    
    # Save final metrics
    if trainer.state.eval_metrics:
        final_acc = trainer.state.eval_metrics.get('MulticlassAccuracy', {}).get('val', 0)
        print(f"Final validation accuracy: {final_acc:.4f}")


if __name__ == '__main__':
    main()
