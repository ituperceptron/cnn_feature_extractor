from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
from ..utils.transforms import get_data_augmentation_transforms

class CustomImageDataset(Dataset):
    """Dataset for loading custom image classification data."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Get class names from subdirectories
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Get all image paths and labels
        self.samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_custom_dataset(data_dir, batch_size=32, num_workers=4, image_size=224, augment=False):
    """Load a custom image dataset from directory.
    
    Args:
        data_dir: Path to dataset directory containing 'train' and 'val' subdirs
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        image_size: Size to which images will be resized
        augment: Boolean to apply data augmentation
        
    Returns:
        train_loader, val_loader, num_classes
    """
    data_dir = Path(data_dir)
    if not (data_dir / 'train').exists() or not (data_dir / 'val').exists():
        raise ValueError("Dataset directory must contain 'train' and 'val' subdirectories")
    
    # Get transforms
    if augment:
        transform = get_data_augmentation_transforms(image_size=image_size)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Create datasets
    train_dataset = CustomImageDataset(data_dir / 'train', transform=transform)
    val_dataset = CustomImageDataset(data_dir / 'val', transform=transform)
    
    # Verify classes match between train and val
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation sets have different classes")
    
    # Create data loaders
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
    
    return train_loader, val_loader, len(train_dataset.classes) 