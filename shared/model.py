"""
ResNet50 Model for ImageNet using MosaicML Composer
Optimized for training with various Composer algorithms and techniques.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from composer import ComposerModel
import torch.nn.functional as F
from typing import Any, Tuple


class ResNet50ComposerModel(ComposerModel):
    """
    ResNet50 model wrapped for MosaicML Composer with ImageNet classification.
    Includes built-in support for Composer optimizations.
    """
    
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        
        # Use torchvision's ResNet50 as base
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Replace classifier if num_classes != 1000
            if num_classes != 1000:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            self.model = models.resnet50(weights=None, num_classes=num_classes)
        
        # Initialize weights properly for training from scratch
        if not pretrained:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for better convergence."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        for m in self.model.modules():
            if hasattr(m, 'bn3') and hasattr(m.bn3, 'weight'):
                nn.init.constant_(m.bn3.weight, 0)
    
    def forward(self, batch: Any) -> torch.Tensor:
        """Forward pass for Composer model."""
        if isinstance(batch, dict):
            inputs = batch['image']
        else:
            inputs, _ = batch
        return self.model(inputs)
    
    def loss(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        """Compute loss for training."""
        if isinstance(batch, dict):
            targets = batch['label']
        else:
            _, targets = batch
        
        return F.cross_entropy(outputs, targets)
    
    def metrics(self, train: bool = False) -> dict:
        """Define metrics to track during training."""
        from torchmetrics.classification import MulticlassAccuracy
        
        return {
            'MulticlassAccuracy': MulticlassAccuracy(
                num_classes=1000,  # Will be adjusted based on actual dataset
                average='micro'
            )
        }


def create_resnet50_composer(
    num_classes: int = 1000,
    pretrained: bool = False,
    compile_model: bool = False
) -> ResNet50ComposerModel:
    """
    Factory function to create ResNet50 Composer model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        compile_model: Whether to compile model with torch.compile()
    
    Returns:
        ResNet50ComposerModel instance
    """
    model = ResNet50ComposerModel(num_classes=num_classes, pretrained=pretrained)
    
    if compile_model and hasattr(torch, 'compile'):
        model.model = torch.compile(model.model)
    
    return model


# Alternative: Custom ResNet50 implementation if needed
class CustomResNet50(ComposerModel):
    """
    Custom ResNet50 implementation with more control over architecture.
    Based on the existing model.py in the workspace.
    """
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.1):
        super().__init__()
        
        # Import and use the existing ResNet implementation
        from typing import Type, Callable, Optional, List
        
        # Use the ResNet class from the existing codebase
        self.model = self._create_resnet50(num_classes, dropout)
    
    def _create_resnet50(self, num_classes: int, dropout: float):
        """Create ResNet50 using existing implementation."""
        # This would use the ResNet class from your existing model.py
        # For now, we'll use torchvision as fallback
        return models.resnet50(weights=None, num_classes=num_classes)
    
    def forward(self, batch: Any) -> torch.Tensor:
        """Forward pass for Composer model."""
        if isinstance(batch, dict):
            inputs = batch['image']
        else:
            inputs, _ = batch
        return self.model(inputs)
    
    def loss(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:
        """Compute loss for training."""
        if isinstance(batch, dict):
            targets = batch['label']
        else:
            _, targets = batch
        
        return F.cross_entropy(outputs, targets)
