"""
Evaluation script for trained ResNet50 models.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from model import create_resnet50_composer
from data_utils import create_dataloaders
from utils import log_system_info, check_gpu_memory


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int = 1000
) -> dict:
    """Evaluate model on given dataloader."""
    model.eval()
    
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    total_loss = 0.0
    
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model((inputs, targets))
            loss = F.cross_entropy(outputs, targets)
            
            # Calculate accuracy
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top1 = pred_top5[:, 0]
            
            correct_top1 += pred_top1.eq(targets).sum().item()
            correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Store predictions for detailed analysis
            all_predictions.extend(pred_top1.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate final metrics
    top1_accuracy = correct_top1 / total_samples
    top5_accuracy = correct_top5 / total_samples
    avg_loss = total_loss / total_samples
    
    results = {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'average_loss': avg_loss,
        'total_samples': total_samples,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return results


def analyze_per_class_accuracy(predictions: list, targets: list, num_classes: int = 1000):
    """Analyze per-class accuracy."""
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    for i in range(num_classes):
        class_mask = targets == i
        if class_mask.sum() > 0:
            class_total[i] = class_mask.sum()
            class_correct[i] = (predictions[class_mask] == i).sum()
    
    # Calculate per-class accuracy
    class_accuracy = np.divide(
        class_correct, 
        class_total, 
        out=np.zeros_like(class_correct), 
        where=class_total != 0
    )
    
    return {
        'per_class_accuracy': class_accuracy.tolist(),
        'mean_class_accuracy': class_accuracy[class_total > 0].mean(),
        'worst_classes': np.argsort(class_accuracy)[:10].tolist(),
        'best_classes': np.argsort(class_accuracy)[-10:].tolist()
    }


def setup_args():
    """Setup evaluation arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ResNet50 on ImageNet')
    
    # Model arguments
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', default='torchvision', choices=['torchvision', 'custom'])
    
    # Data arguments
    parser.add_argument('--data-subset', default='full', help='Dataset subset to evaluate on')
    parser.add_argument('--batch-size', type=int, default=256, help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=8, help='Data loading workers')
    parser.add_argument('--use-hf', action='store_true', default=True, help='Use HuggingFace dataset')
    
    # Evaluation arguments
    parser.add_argument('--device', default='auto', help='Device for evaluation')
    parser.add_argument('--precision', default='fp32', choices=['fp32', 'fp16'], help='Evaluation precision')
    parser.add_argument('--save-results', help='Path to save evaluation results')
    parser.add_argument('--analyze-classes', action='store_true', help='Perform per-class analysis')
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: str):
    """Load model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Load checkpoint
    if device == 'cpu':
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Extract model state dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if present (from Composer checkpoints)
    if any(key.startswith('model.') for key in state_dict.keys()):
        state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    
    # Load state dict
    model.model.load_state_dict(state_dict, strict=True)
    print("✅ Checkpoint loaded successfully!")
    
    # Print checkpoint info if available
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'best_accuracy' in checkpoint:
            print(f"Best accuracy: {checkpoint['best_accuracy']:.4f}")


def main():
    """Main evaluation function."""
    args = setup_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    log_system_info()
    
    # Create model
    print("Creating model...")
    model = create_resnet50_composer(num_classes=1000, pretrained=False)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.checkpoint, model, device)
    
    # Set precision
    if args.precision == 'fp16' and device == 'cuda':
        model = model.half()
        print("Using FP16 precision")
    
    # Create data loader (only validation set)
    print("Loading validation dataset...")
    _, val_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=None if args.data_subset == 'full' else int(args.data_subset),
        use_hf=args.use_hf
    )
    
    # Check GPU memory before evaluation
    if device == 'cuda':
        check_gpu_memory()
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluate_model(model, val_loader, device)
    
    # Print results
    print("\n📊 Evaluation Results:")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    print(f"Average Loss: {results['average_loss']:.4f}")
    print(f"Total Samples: {results['total_samples']:,}")
    
    # Per-class analysis
    if args.analyze_classes:
        print("\n🔍 Per-class analysis...")
        class_results = analyze_per_class_accuracy(
            results['predictions'], 
            results['targets']
        )
        print(f"Mean Class Accuracy: {class_results['mean_class_accuracy']:.4f}")
        print(f"Worst 5 classes: {class_results['worst_classes'][:5]}")
        print(f"Best 5 classes: {class_results['best_classes'][-5:]}")
        
        results.update(class_results)
    
    # Save results
    if args.save_results:
        # Remove large arrays before saving
        save_results = {k: v for k, v in results.items() 
                      if k not in ['predictions', 'targets', 'per_class_accuracy']}
        
        with open(args.save_results, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\n💾 Results saved to {args.save_results}")
    
    # GPU memory check after evaluation
    if device == 'cuda':
        print("\nFinal GPU memory usage:")
        check_gpu_memory()
    
    # Check if we met the target accuracy
    target_accuracy = 0.78
    if results['top1_accuracy'] >= target_accuracy:
        print(f"\n🎉 SUCCESS! Achieved target accuracy of {target_accuracy:.1%}")
        print(f"Final accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
    else:
        print(f"\n⚠️  Target accuracy of {target_accuracy:.1%} not reached")
        print(f"Current accuracy: {results['top1_accuracy']:.4f} ({results['top1_accuracy']*100:.2f}%)")
        gap = target_accuracy - results['top1_accuracy']
        print(f"Gap to target: {gap:.4f} ({gap*100:.2f} percentage points)")


if __name__ == '__main__':
    main()
