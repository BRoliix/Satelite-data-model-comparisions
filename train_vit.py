"""Training script for Vision Transformer models on satellite imagery."""

import torch
import argparse
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import get_dataloaders
from utils.trainer import Trainer
from models.vit_models import get_vit_model


def train_vit(args):
    """Train Vision Transformer model."""
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data loaders
    print(f"\nLoading {args.dataset} dataset...")
    dataloaders = get_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Model
    print(f"\nLoading {args.model} model...")
    model = get_vit_model(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Trainer
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training
    print(f"\nStarting training for {args.epochs} epochs...")
    trainer.train(
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=args.epochs,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Save history
    history_path = checkpoint_dir / 'training_history.json'
    trainer.save_history(str(history_path))
    print(f"\nTraining history saved to {history_path}")
    
    # Test
    print("\nTesting on test set...")
    test_results = trainer.test(dataloaders['test'])
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_results['accuracy']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall:    {test_results['recall']:.4f}")
    print(f"  F1-Score:  {test_results['f1']:.4f}")
    
    # Save results
    results_path = checkpoint_dir / 'test_results.json'
    import json
    results_to_save = {
        'accuracy': float(test_results['accuracy']),
        'precision': float(test_results['precision']),
        'recall': float(test_results['recall']),
        'f1': float(test_results['f1']),
    }
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ViT for satellite image classification')
    
    # Data
    parser.add_argument('--dataset', default='EuroSAT', choices=['EuroSAT', 'BigEarthNet'],
                       help='Dataset name')
    parser.add_argument('--data-dir', default='./data/EuroSAT',
                       help='Path to dataset')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    
    # Model
    parser.add_argument('--model', default='vit_base',
                       choices=['vit_base', 'vit_large', 'vit_small',
                               'deit_base', 'swin_base', 'swin_tiny', 
                               'beit_base', 'coatnet_1'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate (ViTs typically use lower LR)')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step'],
                       help='Learning rate scheduler')
    
    # Other
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    train_vit(args)
