"""
Expected dataset structure:
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
"""

from cnn_feature_extractor import CNNFeatureExtractor
from cnn_feature_extractor.utils.dataset import load_custom_dataset
import torch
import os

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set desired image size
    image_size = 224  # Standard size for most CNN models
    
    # Load custom dataset with data augmentation
    train_loader, val_loader, num_classes = load_custom_dataset(
        data_dir=r"data\bt_veri_seti",
        batch_size=32,
        num_workers=2,  # Reduced number of workers as suggested by warning
        image_size=image_size,
        augment=True  # Enable data augmentation
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Ensure model saving directories exist
    os.makedirs('ml_models', exist_ok=True)
    os.makedirs('cnn_models', exist_ok=True)
    
    # Initialize feature extractor
    extractor = CNNFeatureExtractor(verbose=True)
    
    # Run feature extraction and ML comparison
    results = extractor.fit(
        train_loader, 
        val_loader, 
        cnn_models=['efficientnet_b0'],  # Try specific CNN models
        ml_models=['LogisticRegression']  # Try specific ML models
    )
    
    print("\nModels have been saved to:")
    print(f"- CNN models: {os.path.abspath('cnn_models')}")
    print(f"- ML models: {os.path.abspath('ml_models')}") 