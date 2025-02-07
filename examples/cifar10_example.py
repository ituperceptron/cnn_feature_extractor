"""
Example of using CNNFeatureExtractor with CIFAR-10 dataset.
"""

import torch
import torchvision
from torchvision import transforms
from cnn_feature_extractor import CNNFeatureExtractor

def main():
    # Set desired image size
    image_size = 128

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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

    # Initialize feature extractor
    extractor = CNNFeatureExtractor(save_path='cifar10_results.csv')
    
    # 1. Tiny models only
    #results_tiny = extractor.fit(train_loader, val_loader, cnn_models='tiny')
    
    # 2. Specific models
    results_mix = extractor.fit(train_loader, val_loader, 
                              cnn_models=['resnet18', 'efficientnet_b0'],
                              ml_models=['LogisticRegression'])

if __name__ == '__main__':
    main() 