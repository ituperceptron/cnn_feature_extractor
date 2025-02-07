"""CNN model packages based on model size."""

CNN_PACKAGES = {
    'tiny': [
        'mobilenet_v2',
        'mobilenet_v3_small',
        'efficientnet_b0',
        'convnext_tiny',
        'resnet18'
    ],
    'small': [
        'resnet34',
        'densenet121',
        'mobilenet_v3_large',
        'efficientnet_b1',
        'convnext_small'
    ],
    'medium': [
        'resnet50',
        'densenet169',
        'vgg16',
        'efficientnet_b2',
        'convnext_base'
    ],
    'large': [
        'resnet101',
        'densenet201',
        'vgg19',
        'efficientnet_b3',
        'convnext_large'
    ],
    'biggest': [
        'resnet152',
        'densenet201',
        'efficientnet_b7',
        'convnext_large',
        'vgg19'
    ]
}

def get_cnn_models(package_or_models):
    """Get CNN models based on package name or specific models list.
    
    Args:
        package_or_models: Can be:
            - A string (package name)
            - A list of strings (specific models or package names)
            - A string with '+' (combination of packages/models)
    """
    # If it's already a list, return it
    if isinstance(package_or_models, list):
        return package_or_models
        
    # If it's a string with '+', split and process each part
    if isinstance(package_or_models, str) and '+' in package_or_models:
        models = set()  # Use set to avoid duplicates
        parts = [p.strip() for p in package_or_models.split('+')]
        
        for part in parts:
            # If it's a package name
            if part in CNN_PACKAGES:
                models.update(CNN_PACKAGES[part])
            # If it's a specific model
            else:
                models.add(part)
                
        return list(models)
    
    # If it's a single package name
    if package_or_models in CNN_PACKAGES:
        return CNN_PACKAGES[package_or_models]
        
    # If it's a single model name
    return [package_or_models]

def list_packages():
    """Print available CNN packages and their models."""
    print("\n=== Available CNN Packages ===")
    for package, models in CNN_PACKAGES.items():
        print(f"\n{package.upper()} Package:")
        for model in models:
            print(f"  • {model}")
    
    print("\nYou can combine packages and models with '+':")
    print("Examples:")
    print("  • cnn_models='tiny + medium'")
    print("  • cnn_models='tiny + densenet201'")
    print("  • cnn_models='biggest + resnet18'") 