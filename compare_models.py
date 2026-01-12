"""Comparison script for CNN vs ViT models."""

import torch
import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import get_dataloaders
from utils.trainer import Trainer
from utils.comparison import ModelComparator, plot_training_history, plot_confusion_matrix
from utils.comparison import compare_model_complexity
from models.cnn_models import get_model as get_cnn_model
from models.vit_models import get_vit_model
from tqdm import tqdm


def compare_models(args):
    """Compare CNN and ViT models."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    dataloaders = get_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    # Models to compare
    cnn_models = ['resnet50', 'resnet101', 'efficientnet_b4', 'densenet121']
    vit_models = ['vit_base', 'vit_small', 'deit_base', 'swin_base']
    
    all_models = cnn_models + vit_models if args.compare_all else (
        cnn_models if args.models_type == 'cnn' else vit_models
    )
    
    comparator = ModelComparator()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and evaluate each model
    for model_name in all_models:
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        # Load model
        if model_name in cnn_models:
            model = get_cnn_model(
                model_name=model_name,
                num_classes=args.num_classes,
                pretrained=args.pretrained
            )
        else:
            model = get_vit_model(
                model_name=model_name,
                num_classes=args.num_classes,
                pretrained=args.pretrained
            )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        
        # Trainer
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            scheduler_type=args.scheduler
        )
        
        # Create checkpoint directory
        checkpoint_dir = results_dir / f"checkpoints/{model_name}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training
        trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            num_epochs=args.epochs,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        # Save training history
        history_path = checkpoint_dir / 'training_history.json'
        trainer.save_history(str(history_path))
        
        # Plot training history
        plot_training_history(
            trainer.train_history,
            save_path=results_dir / f"training_history_{model_name}.png"
        )
        plot_training_history(
            trainer.val_history,
            save_path=results_dir / f"validation_history_{model_name}.png"
        )
        
        # Test
        print(f"\nTesting {model_name}...")
        test_results = trainer.test(dataloaders['test'])
        
        print(f"Test Results for {model_name}:")
        print(f"  Accuracy:  {test_results['accuracy']:.4f}")
        print(f"  Precision: {test_results['precision']:.4f}")
        print(f"  Recall:    {test_results['recall']:.4f}")
        print(f"  F1-Score:  {test_results['f1']:.4f}")
        
        # Add to comparator
        comparator.add_result(model_name, test_results)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            test_results['confusion_matrix'],
            save_path=results_dir / f"confusion_matrix_{model_name}.png"
        )
    
    # Save comparison results
    print(f"\n{'='*60}")
    print("Saving comparison results...")
    print(f"{'='*60}")
    
    comparator.save_results(results_dir / 'comparison_results.json')
    comparator.plot_comparison(
        save_path=results_dir / 'model_comparison.png',
        metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    comparator.print_summary()
    
    # Create final report
    from utils.comparison import create_comparison_report
    create_comparison_report(comparator.results, 
                            results_dir / 'comparison_report.txt')
    
    print(f"\nAll results saved to {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare CNN and ViT models')
    
    # Data
    parser.add_argument('--dataset', default='EuroSAT', choices=['EuroSAT', 'BigEarthNet'],
                       help='Dataset name')
    parser.add_argument('--data-dir', default='./data/EuroSAT',
                       help='Path to dataset')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    
    # Model selection
    parser.add_argument('--compare-all', action='store_true', default=False,
                       help='Compare all CNN and ViT models')
    parser.add_argument('--models-type', default='cnn', choices=['cnn', 'vit'],
                       help='Type of models to compare (if --compare-all is False)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (reduced for quick testing)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'step'],
                       help='Learning rate scheduler')
    
    # Other
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--results-dir', default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    compare_models(args)
