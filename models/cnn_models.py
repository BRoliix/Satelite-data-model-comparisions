"""CNN baseline models for satellite image classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50(nn.Module):
    """ResNet50 baseline for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class ResNet101(nn.Module):
    """ResNet101 baseline for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.resnet101(pretrained=pretrained)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class EfficientNetB4(nn.Module):
    """EfficientNetB4 for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Replace final classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class DenseNet121(nn.Module):
    """DenseNet121 for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
        
        # Replace final classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class SimpleCNN(nn.Module):
    """Simple custom CNN baseline for satellite image classification."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def get_model(model_name, num_classes=10, pretrained=True):
    """Factory function to get CNN models.
    
    Args:
        model_name: Name of the model ('resnet50', 'resnet101', 'efficientnet_b4', 
                    'densenet121', 'simple_cnn')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    
    models_dict = {
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'efficientnet_b4': EfficientNetB4,
        'densenet121': DenseNet121,
        'simple_cnn': SimpleCNN
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_cls = models_dict[model_name]
    
    if model_name == 'simple_cnn':
        return model_cls(num_classes=num_classes)
    else:
        return model_cls(num_classes=num_classes, pretrained=pretrained)
