import torch
import torch.nn as nn
from torchvision import models

class CNNFeatureExtractor:
    def __init__(self, model_name='resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._get_model(model_name)
        
    def _get_model(self, model_name: str) -> nn.Module:
        """Get a pre-trained CNN model without its classification head."""
        model = getattr(models, model_name)(pretrained=True)
        
        # Remove the final classification layer
        if hasattr(model, 'fc'):  # ResNet, DenseNet
            features = nn.Sequential(*list(model.children())[:-1])
        elif hasattr(model, 'classifier'):  # VGG, MobileNet
            features = model.features
        else:
            features = nn.Sequential(*list(model.children())[:-1])
            
        return features.to(self.device).eval()
        
    def extract_features(self, images):
        """Extract features from images."""
        with torch.no_grad():
            features = self.model(images.to(self.device))
            features = features.reshape(features.size(0), -1)
        return features 