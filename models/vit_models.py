"""Vision Transformer models for satellite image classification."""

import torch
import torch.nn as nn
import timm
from einops import rearrange


class ViT_Base(nn.Module):
    """Vision Transformer Base model for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True, img_size=224, patch_size=16):
        super().__init__()
        
        model_name = 'vit_base_patch16_224'
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class ViT_Large(nn.Module):
    """Vision Transformer Large model for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True, img_size=224, patch_size=16):
        super().__init__()
        
        model_name = 'vit_large_patch16_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class ViT_Small(nn.Module):
    """Vision Transformer Small model for satellite image classification."""
    
    def __init__(self, num_classes=10, pretrained=True, img_size=224, patch_size=16):
        super().__init__()
        
        model_name = 'vit_small_patch16_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class DeiT_Base(nn.Module):
    """Data-efficient Image Transformers Base model."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        model_name = 'deit_base_patch16_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class Swin_Base(nn.Module):
    """Swin Transformer Base model."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        model_name = 'swin_base_patch4_window7_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class Swin_Tiny(nn.Module):
    """Swin Transformer Tiny model."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        model_name = 'swin_tiny_patch4_window7_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class BEiT_Base(nn.Module):
    """BERT pre-training of Image Transformers Base model."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        model_name = 'beit_base_patch16_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class CoAtNet_1(nn.Module):
    """Hybrid CNN-Transformer model."""
    
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        model_name = 'coatnet_1_rw_224'
        self.model = timm.create_model(model_name, pretrained=pretrained,
                                       num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


def get_vit_model(model_name, num_classes=10, pretrained=True):
    """Factory function to get Vision Transformer models.
    
    Args:
        model_name: Name of the model ('vit_base', 'vit_large', 'vit_small',
                    'deit_base', 'swin_base', 'swin_tiny', 'beit_base', 'coatnet_1')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    
    models_dict = {
        'vit_base': ViT_Base,
        'vit_large': ViT_Large,
        'vit_small': ViT_Small,
        'deit_base': DeiT_Base,
        'swin_base': Swin_Base,
        'swin_tiny': Swin_Tiny,
        'beit_base': BEiT_Base,
        'coatnet_1': CoAtNet_1,
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_cls = models_dict[model_name]
    return model_cls(num_classes=num_classes, pretrained=pretrained)
