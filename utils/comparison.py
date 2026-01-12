"""Utilities for model comparison and analysis."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pandas as pd


class ModelComparator:
    """Compare multiple models' performance."""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name, test_metrics):
        """Add test results for a model.
        
        Args:
            model_name: Name of the model
            test_metrics: Dictionary with metrics from Trainer.test()
        """
        self.results[model_name] = test_metrics
    
    def save_results(self, path):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for model_name, metrics in self.results.items():
            results_json[model_name] = {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'confusion_matrix': metrics['confusion_matrix'].tolist(),
            }
        
        with open(path, 'w') as f:
            json.dump(results_json, f, indent=2)
    
    def plot_comparison(self, save_path='./results/comparison.png', 
                       metrics=['accuracy', 'precision', 'recall', 'f1']):
        """Plot comparison of models.
        
        Args:
            save_path: Path to save the plot
            metrics: List of metrics to plot
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True)
        
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            values = [self.results[name][metric] for name in model_names]
            
            axes[idx].bar(model_names, values, color='steelblue')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f'{metric.capitalize()} Comparison')
            axes[idx].set_ylim([0, 1])
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
        plt.close()
    
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        
        # Create summary table
        summary_data = {}
        for model_name, metrics in self.results.items():
            summary_data[model_name] = {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
            }
        
        df = pd.DataFrame(summary_data).T
        print(df)
        print("="*80 + "\n")


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Plot confusion matrix.
    
    Args:
        cm: Confusion matrix from sklearn
        class_names: List of class names
        save_path: Path to save the plot
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training history.
    
    Args:
        history: Dictionary with 'train' and 'val' keys containing lists of losses/accuracies
        save_path: Path to save the plot
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    if 'loss' in history['train'] and 'loss' in history['val']:
        axes[0].plot(history['train']['loss'], label='Train Loss')
        axes[0].plot(history['val']['loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in history['train'] and 'accuracy' in history['val']:
        axes[1].plot(history['train']['accuracy'], label='Train Accuracy')
        axes[1].plot(history['val']['accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.close()


def create_comparison_report(results_dict, save_path='./results/comparison_report.txt'):
    """Create a comprehensive comparison report.
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the report
    """
    
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SATELLITE IMAGE CLASSIFICATION - MODEL COMPARISON REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model Name':<30} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}\n")
        f.write("-"*80 + "\n")
        
        for model_name, metrics in results_dict.items():
            f.write(f"{model_name:<30} {metrics['accuracy']:<15.4f} {metrics['precision']:<15.4f} "
                   f"{metrics['recall']:<15.4f} {metrics['f1']:<15.4f}\n")
        
        f.write("-"*80 + "\n\n")
        
        # Detailed analysis
        f.write("DETAILED ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        for model_name, metrics in results_dict.items():
            f.write(f"\n{model_name}\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Comparison report saved to {save_path}")


def calculate_model_complexity(model):
    """Calculate model complexity metrics.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with complexity metrics
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
    }


def compare_model_complexity(models_dict, save_path=None):
    """Compare complexity of multiple models.
    
    Args:
        models_dict: Dictionary with {model_name: model}
        save_path: Path to save the plot
    """
    
    complexity_data = {}
    
    for model_name, model in models_dict.items():
        complexity = calculate_model_complexity(model)
        complexity_data[model_name] = complexity
    
    # Plot parameter counts
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    model_names = list(complexity_data.keys())
    total_params = [complexity_data[name]['total_parameters'] / 1e6 for name in model_names]
    trainable_params = [complexity_data[name]['trainable_parameters'] / 1e6 for name in model_names]
    
    # Total parameters
    axes[0].bar(model_names, total_params, color='steelblue')
    axes[0].set_ylabel('Parameters (Millions)')
    axes[0].set_title('Total Parameters')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Trainable parameters
    axes[1].bar(model_names, trainable_params, color='darkgreen')
    axes[1].set_ylabel('Parameters (Millions)')
    axes[1].set_title('Trainable Parameters')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return complexity_data
