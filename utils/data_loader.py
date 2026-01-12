"""Data loading and preprocessing utilities for satellite imagery."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import requests
from pathlib import Path
from tqdm import tqdm
import json


class EuroSATDataset(Dataset):
    """EuroSAT dataset wrapper for satellite image classification.
    
    Dataset Info:
    - 27,000 labeled Sentinel-2 satellite images
    - 64x64 pixels, 13 spectral bands
    - 10 land-use classes
    - Classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial, 
               Pasture, Permanent Crop, Residential, River, Sea/Lake
    """
    
    CLASSES = [
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
    
    def __init__(self, root_dir, split='train', transform=None, download=False):
        """
        Args:
            root_dir: Root directory for dataset
            split: 'train', 'val', or 'test'
            transform: Torchvision transforms
            download: Whether to download dataset
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if download:
            self._download_dataset()
        
        self._load_dataset()
    
    def _download_dataset(self):
        """Download EuroSAT dataset from source."""
        # Note: This is a placeholder. In practice, you would download from:
        # http://madm.web.unc.edu/sentinel2/
        print("Dataset should be downloaded manually from:")
        print("http://madm.web.unc.edu/sentinel2/")
        print("Extract to: data/EuroSAT/")
    
    def _load_dataset(self):
        """Load dataset from directory structure."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        for idx, class_name in enumerate(self.CLASSES):
            class_dir = split_dir / class_name
            if class_dir.exists():
                # Try both .tif and .jpg formats
                for img_path in sorted(list(class_dir.glob('*.tif')) + list(class_dir.glob('*.jpg'))):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {split_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image - handle both TIF and JPG
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            # Return a blank image as fallback
            image = Image.new('RGB', (64, 64))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class BigEarthNetDataset(Dataset):
    """BigEarthNet dataset wrapper for satellite image classification.
    
    Dataset Info:
    - 590,326 labeled Sentinel-1 & Sentinel-2 pairs
    - 120x120 pixels, multi-spectral bands
    - 43 land-use classes (or 19 for coarse classification)
    - High-resolution RGB imagery
    """
    
    CLASSES_19 = [
        'Urban fabric', 'Industrial or commercial units', 'Arable land (annual crops)',
        'Permanent crops', 'Complex and mixed cultivation patterns', 'Orchards',
        'Forests', 'Grasslands', 'Root crops and tubers', 'Pastures',
        'Complex cultivation patterns', 'Land principally occupied by agriculture',
        'Transitional woodland/shrub', 'Beaches, dunes, sands', 'Bare rock',
        'Transitional woodland-shrub', 'Water', 'Clouds and Shadows', 'No Data'
    ]
    
    def __init__(self, root_dir, split='train', transform=None, 
                 use_coarse_labels=True):
        """
        Args:
            root_dir: Root directory for dataset
            split: 'train', 'val', or 'test'
            transform: Torchvision transforms
            use_coarse_labels: Use 19 coarse classes or 43 fine classes
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_coarse_labels = use_coarse_labels
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load BigEarthNet dataset from directory structure."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        for patch_dir in sorted(split_dir.iterdir()):
            if patch_dir.is_dir():
                metadata_file = patch_dir / 'metadata.json'
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    labels = (metadata.get('labels_coarse_id') if self.use_coarse_labels 
                             else metadata.get('labels_id', []))
                    
                    if labels:
                        # Use first label for single-label classification
                        label = labels[0]
                        img_path = patch_dir / 'TCI.tif'  # True Color Image
                        
                        if img_path.exists():
                            self.image_paths.append(img_path)
                            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, augment=True):
    """Get data transforms for training and evaluation.
    
    Args:
        img_size: Target image size
        augment: Whether to apply data augmentation
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]) if augment else transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {'train': train_transform, 'val': val_transform}


def get_dataloaders(dataset_name='EuroSAT', root_dir='./data', 
                    batch_size=32, num_workers=4, img_size=224):
    """Create dataloaders for training and evaluation.
    
    Args:
        dataset_name: 'EuroSAT' or 'BigEarthNet'
        root_dir: Path to dataset root
        batch_size: Batch size
        num_workers: Number of data loading workers
        img_size: Image size
    
    Returns:
        Dictionary with train, val, test dataloaders
    """
    
    transforms_dict = get_transforms(img_size=img_size, augment=True)
    
    if dataset_name == 'EuroSAT':
        dataset_cls = EuroSATDataset
    elif dataset_name == 'BigEarthNet':
        dataset_cls = BigEarthNetDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_dataset = dataset_cls(
        root_dir=root_dir,
        split='train',
        transform=transforms_dict['train'],
        download=False
    )
    
    val_dataset = dataset_cls(
        root_dir=root_dir,
        split='val',
        transform=transforms_dict['val'],
        download=False
    )
    
    test_dataset = dataset_cls(
        root_dir=root_dir,
        split='test',
        transform=transforms_dict['val'],
        download=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
