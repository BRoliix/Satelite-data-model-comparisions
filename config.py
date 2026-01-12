"""
Configuration file for satellite image classification project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
UTILS_DIR = PROJECT_ROOT / 'utils'
RESULTS_DIR = PROJECT_ROOT / 'results'
CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset paths
EUROSAT_DATA_DIR = DATA_DIR / 'EuroSAT'
BIGEARTHNET_DATA_DIR = DATA_DIR / 'BigEarthNet'

# Default hyperparameters
DEFAULT_CONFIG = {
    'dataset': {
        'name': 'EuroSAT',
        'num_classes': 10,
        'img_size': 224,
        'num_workers': 4,
    },
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'weight_decay': 1e-4,
    },
    'vit_training': {
        'learning_rate': 1e-4,  # Lower LR for ViTs
        'scheduler': 'cosine',
    },
    'model': {
        'pretrained': True,
        'device': 'cuda',
    }
}

# Model configurations
CNN_MODELS = {
    'resnet50': {
        'params': 23.5e6,
        'input_size': 224,
    },
    'resnet101': {
        'params': 44.5e6,
        'input_size': 224,
    },
    'efficientnet_b4': {
        'params': 17.7e6,
        'input_size': 384,
    },
    'densenet121': {
        'params': 7.9e6,
        'input_size': 224,
    },
    'simple_cnn': {
        'params': 2.3e6,
        'input_size': 224,
    }
}

VIT_MODELS = {
    'vit_base': {
        'params': 86.6e6,
        'input_size': 224,
    },
    'vit_large': {
        'params': 304.3e6,
        'input_size': 224,
    },
    'vit_small': {
        'params': 22.1e6,
        'input_size': 224,
    },
    'deit_base': {
        'params': 86.6e6,
        'input_size': 224,
    },
    'swin_base': {
        'params': 87.8e6,
        'input_size': 224,
    },
    'swin_tiny': {
        'params': 28.3e6,
        'input_size': 224,
    },
    'beit_base': {
        'params': 86.6e6,
        'input_size': 224,
    },
    'coatnet_1': {
        'params': 41.7e6,
        'input_size': 224,
    }
}

# EuroSAT classes
EUROSAT_CLASSES = [
    'Annual Crop',
    'Forest',
    'Herbaceous Vegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'Permanent Crop',
    'Residential',
    'River',
    'Sea/Lake'
]

# BigEarthNet coarse classes (19)
BIGEARTHNET_CLASSES_COARSE = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land (annual crops)',
    'Permanent crops',
    'Complex and mixed cultivation patterns',
    'Orchards',
    'Forests',
    'Grasslands',
    'Root crops and tubers',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Bare rock',
    'Transitional woodland-shrub',
    'Water',
    'Clouds and Shadows',
    'No Data'
]

# Normalization values (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Sentinel-2 normalization (optional for multi-spectral data)
SENTINEL2_MEAN = [0.40736, 0.38892, 0.36621]
SENTINEL2_STD = [0.21862, 0.16849, 0.13433]
