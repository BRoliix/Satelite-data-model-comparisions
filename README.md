# Satellite Image Classification: ViT vs CNN

A comprehensive deep learning project comparing Vision Transformers (ViT) with traditional Convolutional Neural Networks (CNNs) for fine-grained land-use classification using satellite imagery.

## Project Overview

This project evaluates the effectiveness of Vision Transformers on satellite image classification tasks, specifically land-use classification. We compare multiple state-of-the-art architectures:

### CNN Models
- **ResNet50 & ResNet101** - Deep residual networks with skip connections
- **EfficientNetB4** - Efficient network scaling across depth, width, and resolution
- **DenseNet121** - Dense connections for feature reuse
- **SimpleCNN** - Custom baseline architecture

### Vision Transformer Models
- **ViT (Base, Small, Large)** - Standard Vision Transformers with different scales
- **DeiT (Data-efficient Image Transformers)** - Knowledge-distilled variant
- **Swin Transformer** - Hierarchical transformers with local windows
- **BEiT** - BERT pre-training for image transformers
- **CoAtNet** - Hybrid CNN-Transformer model

## Datasets

### EuroSAT
- **27,000** labeled Sentinel-2 satellite images
- **64×64 pixels** with 13 spectral bands (reduced to RGB for compatibility)
- **10 land-use classes**:
  - Annual Crop, Forest, Herbaceous Vegetation
  - Highway, Industrial, Pasture
  - Permanent Crop, Residential, River
  - Sea/Lake
- Source: http://madm.web.unc.edu/sentinel2/

### BigEarthNet
- **590,326** labeled Sentinel-1 & Sentinel-2 pairs
- **120×120 pixels** multi-spectral imagery
- **43 fine-grained or 19 coarse land-use classes**
- High-resolution RGB True Color Images
- Download: https://bigEarthnet.org/

## Project Structure

```
sati/
├── data/                      # Dataset storage
│   ├── EuroSAT/              # EuroSAT dataset
│   └── BigEarthNet/          # BigEarthNet dataset
├── models/
│   ├── cnn_models.py         # CNN architectures
│   ├── vit_models.py         # Vision Transformer architectures
│   └── __init__.py
├── utils/
│   ├── data_loader.py        # Dataset loading and preprocessing
│   ├── trainer.py            # Training and evaluation logic
│   ├── comparison.py         # Model comparison utilities
│   └── __init__.py
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Dataset analysis
│   ├── 02_model_training.ipynb          # Training guide
│   └── 03_results_analysis.ipynb        # Results visualization
├── results/                  # Experiment results
│   ├── checkpoints/          # Model checkpoints
│   ├── comparison_results.json
│   ├── model_comparison.png
│   └── comparison_report.txt
├── train_cnn.py             # Training script for CNNs
├── train_vit.py             # Training script for ViTs
├── compare_models.py        # Comparison script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration, recommended)
- macOS with Apple Silicon or x86-64 processor

### Setup

1. Clone the repository:
```bash
cd /Users/nekonyo/sati
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download datasets:
```bash
# For EuroSAT - Download manually from: http://madm.web.unc.edu/sentinel2/
# Extract to: data/EuroSAT/

# For BigEarthNet - Download from: https://bigEarthnet.org/
# Extract to: data/BigEarthNet/
```

## Usage

### 1. Train a Single CNN Model

```bash
python train_cnn.py \
    --model resnet50 \
    --dataset EuroSAT \
    --data-dir ./data/EuroSAT \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-3
```

Available CNN models: `resnet50`, `resnet101`, `efficientnet_b4`, `densenet121`, `simple_cnn`

### 2. Train a Single ViT Model

```bash
python train_vit.py \
    --model vit_base \
    --dataset EuroSAT \
    --data-dir ./data/EuroSAT \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4  # Lower LR for ViTs
```

Available ViT models: `vit_base`, `vit_large`, `vit_small`, `deit_base`, `swin_base`, `swin_tiny`, `beit_base`, `coatnet_1`

### 3. Compare Multiple Models

```bash
# Compare all CNN models
python compare_models.py \
    --models-type cnn \
    --dataset EuroSAT \
    --data-dir ./data/EuroSAT \
    --epochs 50 \
    --results-dir ./results

# Compare all ViT models
python compare_models.py \
    --models-type vit \
    --dataset EuroSAT \
    --data-dir ./data/EuroSAT \
    --epochs 50 \
    --results-dir ./results

# Compare CNN vs ViT
python compare_models.py \
    --compare-all \
    --dataset EuroSAT \
    --data-dir ./data/EuroSAT \
    --epochs 50 \
    --results-dir ./results
```

### 4. Using Jupyter Notebooks

Start Jupyter and navigate to the notebooks:
```bash
jupyter notebook notebooks/
```

## Key Arguments

### Data Arguments
- `--dataset`: Choose between `EuroSAT` or `BigEarthNet`
- `--data-dir`: Path to dataset root directory
- `--num-classes`: Number of classification classes (10 for EuroSAT, 19 or 43 for BigEarthNet)
- `--img-size`: Input image size (default: 224)

### Model Arguments
- `--model`: Model architecture name
- `--pretrained`: Use pretrained weights (default: True)

### Training Arguments
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-3 for CNNs, 1e-4 for ViTs)
- `--optimizer`: Optimizer type (`adam` or `sgd`)
- `--scheduler`: Learning rate scheduler (`cosine` or `step`)

### Other Arguments
- `--num-workers`: Number of data loading workers (default: 4)
- `--checkpoint-dir`: Directory to save model checkpoints
- `--results-dir`: Directory to save results

## Expected Results

### Typical Performance on EuroSAT (100 epochs, 32 batch size)

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| ResNet50 | ~0.94 | ~0.94 | ~0.94 | ~0.94 | 23.5M |
| EfficientNetB4 | ~0.95 | ~0.95 | ~0.95 | ~0.95 | 17.7M |
| ViT-Base | ~0.96 | ~0.96 | ~0.96 | ~0.96 | 86.6M |
| DeiT-Base | ~0.95 | ~0.95 | ~0.95 | ~0.95 | 86.6M |
| Swin-Base | ~0.97 | ~0.97 | ~0.97 | ~0.97 | 87.8M |

*Note: Actual results vary based on dataset split, hyperparameters, and random seed.*

## Key Findings

### ViT Advantages
1. **Global Receptive Field**: Processes entire image patches initially, capturing long-range dependencies
2. **Transfer Learning**: Excellent performance with pretrained ImageNet weights
3. **Scalability**: Scales better with larger datasets
4. **Interpretability**: Attention maps provide explainability

### CNN Advantages
1. **Efficiency**: Fewer parameters and faster inference
2. **Inductive Bias**: Exploits spatial locality and translation invariance
3. **Data Efficiency**: Works well with limited data
4. **Computational Cost**: Lower memory requirements

### Hybrid Approaches
- **Swin Transformers**: Window-based attention reduces complexity while maintaining ViT benefits
- **CoAtNet**: Combines CNN's inductive bias with Transformer's expressiveness

## Visualization and Analysis

The project generates comprehensive visualizations:

1. **Training Curves**: Loss and accuracy over epochs
2. **Confusion Matrices**: Per-class performance analysis
3. **Model Comparison Plots**: Side-by-side metric comparison
4. **Parameter Analysis**: Model complexity comparison
5. **Detailed Reports**: Classification metrics and analysis

See generated files in `./results/`:
- `model_comparison.png`: Performance comparison
- `training_history_*.png`: Training curves for each model
- `confusion_matrix_*.png`: Confusion matrices
- `comparison_report.txt`: Detailed analysis

## Advanced Usage

### Custom Data Loading
```python
from utils.data_loader import EuroSATDataset, get_transforms

# Load custom dataset
transforms = get_transforms(img_size=224, augment=True)
dataset = EuroSATDataset(
    root_dir='./data/EuroSAT',
    split='train',
    transform=transforms['train']
)
```

### Custom Model Training
```python
from utils.trainer import Trainer
from models.cnn_models import get_model

model = get_model('resnet50', num_classes=10, pretrained=True)
trainer = Trainer(model, device='cuda', learning_rate=1e-3)

# Train
trainer.train(train_loader, val_loader, num_epochs=100)

# Test and get detailed metrics
results = trainer.test(test_loader)
```

### Model Comparison
```python
from utils.comparison import ModelComparator

comparator = ModelComparator()
# Add results from each model
comparator.add_result('resnet50', results_dict)
comparator.plot_comparison()
comparator.print_summary()
```

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (or Apple Silicon with 8GB+ unified memory)
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 16GB+ for batch processing
- **Storage**: 50GB+ for datasets and checkpoints

### Minimum
- GPU: 4GB VRAM
- CPU: 4-core processor
- RAM: 8GB
- Storage: 30GB

## References

### Research Papers
- **Vision Transformers**: Dosovitskiy et al., 2021. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- **DeiT**: Touvron et al., 2021. "Training data-efficient image transformers & distillation through attention"
- **Swin Transformers**: Liu et al., 2021. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
- **BEiT**: Bao et al., 2022. "BEiT: BERT Pre-Training of Image Transformers"
- **CoAtNet**: Dai et al., 2021. "CoAtNet: Marrying Convolution and Attention for All Data Sizes"

### Datasets
- **EuroSAT**: Helber et al., 2019. "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"
- **BigEarthNet**: Sumbul et al., 2019. "BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding"

### Related Projects
- https://github.com/satellite-image-deep-learning/techniques
- https://github.com/zhu-xlab/SATIN
- https://github.com/prs-eth/sentinel2-cloud-detector

## Future Enhancements

1. **Multi-Scale Processing**: Pyramid networks for capturing features at multiple scales
2. **Ensemble Methods**: Combining CNN and ViT predictions
3. **Few-Shot Learning**: Performance with limited training data
4. **Domain Adaptation**: Transfer across different satellite sensors
5. **Attention Visualization**: Explainable AI for model decisions
6. **Weakly Supervised Learning**: Learning with partial labels
7. **Continual Learning**: Adapting to new land-use categories
8. **Efficiency Optimization**: Pruning, quantization, and knowledge distillation


## Acknowledgments

- EuroSAT research teams for dataset curation
- PyTorch team for deep learning framework
- timm library for transformer implementations
- Sentinel-2 data from Copernicus program

---

**Happy training! 🛰️🤖**
