"""Inference script for testing trained models on satellite images."""

import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.cnn_models import get_model as get_cnn_model
from models.vit_models import get_vit_model
from utils.data_loader import get_transforms
from config import EUROSAT_CLASSES


def load_model(model_name, num_classes=10, checkpoint_path=None, device='cuda'):
    """Load a model from checkpoint.
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Model in evaluation mode
    """
    
    # Determine model type
    cnn_models = ['resnet50', 'resnet101', 'efficientnet_b4', 'densenet121', 'simple_cnn']
    
    if model_name in cnn_models:
        model = get_cnn_model(model_name, num_classes=num_classes, pretrained=False)
    else:
        model = get_vit_model(model_name, num_classes=num_classes, pretrained=False)
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model = model.to(device).eval()
    return model


def predict_image(model, image_path, img_size=224, device='cuda'):
    """Predict class for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        img_size: Image size
        device: Device
    
    Returns:
        Dictionary with predictions and confidence scores
    """
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = get_transforms(img_size=img_size, augment=False)['val']
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = outputs.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    # Get top-k predictions
    top_k = 5
    top_probs, top_classes = torch.topk(probs[0], top_k)
    
    return {
        'predicted_class': pred_class,
        'predicted_label': EUROSAT_CLASSES[pred_class],
        'confidence': pred_prob,
        'top_k_predictions': [
            {
                'class': EUROSAT_CLASSES[c.item()],
                'probability': p.item()
            }
            for p, c in zip(top_probs, top_classes)
        ]
    }


def predict_batch(model, image_dir, img_size=224, device='cuda'):
    """Predict for all images in a directory.
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        img_size: Image size
        device: Device
    
    Returns:
        List of predictions
    """
    
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    
    image_paths = [
        p for p in image_dir.rglob('*') 
        if p.suffix.lower() in image_extensions
    ]
    
    results = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for image_path in image_paths:
        try:
            prediction = predict_image(model, image_path, img_size, device)
            prediction['image_path'] = str(image_path)
            results.append(prediction)
            print(f"✓ {image_path.name}: {prediction['predicted_label']} "
                  f"({prediction['confidence']:.2%})")
        except Exception as e:
            print(f"✗ Error processing {image_path.name}: {str(e)}")
    
    return results


def print_prediction(result):
    """Print prediction results nicely."""
    
    print(f"\nPrediction Results for: {result['image_path']}")
    print("=" * 60)
    print(f"Predicted Class: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nTop-5 Predictions:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Class':<30} {'Probability':<15}")
    print("-" * 60)
    
    for idx, pred in enumerate(result['top_k_predictions'], 1):
        print(f"{idx:<6} {pred['class']:<30} {pred['probability']:.4f}")
    
    print("=" * 60)


def main(args):
    """Main inference function."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = load_model(
        model_name=args.model,
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint,
        device=device
    )
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting for: {args.image}")
        result = predict_image(model, args.image, args.img_size, device)
        print_prediction(result)
    
    # Batch prediction
    if args.directory:
        print(f"\nPredicting for images in: {args.directory}")
        results = predict_batch(model, args.directory, args.img_size, device)
        
        # Save results
        if args.save_results:
            import json
            save_path = Path(args.save_results)
            save_path.parent.mkdir(exist_ok=True)
            
            with open(save_path, 'w') as f:
                # Convert tensor items to native Python types
                results_json = []
                for r in results:
                    r_copy = r.copy()
                    results_json.append(r_copy)
                json.dump(results_json, f, indent=2)
            
            print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with trained models')
    
    # Model
    parser.add_argument('--model', required=True,
                       help='Model name (e.g., resnet50, vit_base)')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    
    # Input
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--directory', help='Directory of images for batch prediction')
    
    # Options
    parser.add_argument('--img-size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--save-results', 
                       help='Save predictions to JSON file')
    
    args = parser.parse_args()
    
    if not args.image and not args.directory:
        parser.error("Please provide either --image or --directory")
    
    main(args)
