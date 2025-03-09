from torchvision import transforms

def get_default_transform(image_size=224):
    """Get the default transform for images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to fixed dimensions
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_data_augmentation_transforms(image_size=224):
    """Get the transformations for datasets with data augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to fixed dimensions first
        transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
        transforms.RandomRotation(degrees=15),  # Randomly rotate images by up to 15 degrees
        transforms.ColorJitter(brightness=0.2, saturation=0.2),  # Randomly change brightness and saturation
        transforms.RandomGrayscale(p=0.2),  # Randomly convert images to grayscale with a probability of 0.2
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 