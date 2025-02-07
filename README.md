# cnn_feature_extractor

A Python package for automatic CNN feature extraction and ML model comparison. Extract features from images using pre-trained CNN models and evaluate multiple ML classifiers in one go.

## Installation

```bash
pip install cnn_feature_extractor
```

## Quick Start with CIFAR10

```python
import torch
import torchvision
from torchvision import transforms
from cnn_feature_extractor import CNNFeatureExtractor

# Set image size
image_size = 128

# Define transforms
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

val_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Initialize and run feature extraction + ML comparison
extractor = CNNFeatureExtractor(save_path='cifar10_results.csv')
results = extractor.fit(
    train_loader, 
    val_loader,
    cnn_models=['resnet18', 'efficientnet_b0'],
    ml_models=['LogisticRegression']
)
```

## Using Your Custom Dataset

### Required Dataset Structure
```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class1/
    │   └── image5.jpg
    └── class2/
        └── image6.jpg
```

### Custom Dataset Example

```python
from cnn_feature_extractor import CNNFeatureExtractor
from cnn_feature_extractor.utils.dataset import load_custom_dataset

# Set image size and other parameters
image_size = 224  # Standard size for most CNN models
batch_size = 32
num_workers = 4

# Load your custom dataset
train_loader, val_loader, num_classes = load_custom_dataset(
    data_dir='path/to/your/dataset',  # Path to your dataset root directory
    batch_size=batch_size,
    num_workers=num_workers,
    image_size=image_size,
    augment=True  # Enable data augmentation (optional)
)

# Initialize feature extractor
extractor = CNNFeatureExtractor(save_path='results.csv')

# Run feature extraction and ML comparison
results = extractor.fit(
    train_loader, 
    val_loader,
    cnn_models=['resnet18', 'efficientnet_b0'],  # Choose CNN models
    ml_models=['RandomForest', 'LogisticRegression']  # Choose ML models
)

# Results will be saved to 'results.csv'
print(results)
```

## Features

- 🔄 Support for multiple CNN architectures (ResNet, EfficientNet, etc.)
- 🤖 Multiple ML classifiers (LogisticRegression, RandomForest, etc.)
- 📊 Automatic feature extraction and model evaluation
- 🎯 GPU acceleration when available
- 📈 Data augmentation support
- 💾 Results saved to CSV file

## Available Models

### CNN Feature Extractors

#### Tiny Package (Fast, Lower Accuracy)
- mobilenet_v2
- mobilenet_v3_small
- efficientnet_b0
- convnext_tiny
- resnet18

#### Small Package
- resnet34
- densenet121
- mobilenet_v3_large
- efficientnet_b1
- convnext_small

#### Medium Package
- resnet50
- densenet169
- vgg16
- efficientnet_b2
- convnext_base

#### Large Package
- resnet101
- densenet201
- vgg19
- efficientnet_b3
- convnext_large

#### Biggest Package (Slow, Higher Accuracy)
- resnet152
- densenet201
- efficientnet_b7
- convnext_large
- vgg19

### ML Classifiers
- RandomForest
- SVM (with probability estimation)
- LogisticRegression
- GradientBoosting
- XGBoost
- LightGBM
- KNN
- DecisionTree
- AdaBoost
- GaussianNB
- RidgeClassifier
- SGDClassifier
- LinearSVC

## Package Usage Tips

1. **Choosing CNN Models**:
   - Start with 'tiny' package models for quick experiments
   - Use 'biggest' package models for maximum accuracy
   - Mix models from different packages: `cnn_models=['resnet18', 'efficientnet_b7']`

2. **Choosing ML Models**:
   - Start with fast models like LogisticRegression
   - Use RandomForest or XGBoost for better accuracy
   - Try multiple models: `ml_models=['LogisticRegression', 'RandomForest', 'XGBoost']`

3. **Data Augmentation**:
   - Enable with `augment=True` in `load_custom_dataset`
   - Helps prevent overfitting
   - Especially useful for small datasets

4. **GPU Usage**:
   - GPU is automatically used if available
   - CNN feature extraction is significantly faster on GPU
   - Some ML models (XGBoost, LightGBM) can also use GPU

## Author

**ITU Perceptron**
- Email: ituperceptron@gmail.com

## License

MIT License 