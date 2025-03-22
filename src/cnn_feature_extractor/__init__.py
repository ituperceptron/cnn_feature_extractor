from .extractors.cnn_extractor import CNNFeatureExtractor as BaseCNNExtractor
from .models.classifiers import get_classifiers
from .models.cnn_packages import get_cnn_models, list_packages
from .utils.transforms import get_default_transform
from .utils.metrics import MetricsTracker
from .utils.dataset import load_custom_dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm
import torch
from torchvision import models
import time
import os
import pickle
import joblib

class CNNFeatureExtractor:
    """Automatic CNN feature extraction and ML model comparison."""
    
    def __init__(self, verbose=True, ignore_warnings=True, save_path='results.csv', models_dir='saved_models'):
        if ignore_warnings:
            import warnings
            warnings.filterwarnings('ignore')
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifiers = get_classifiers()
        self.metrics = MetricsTracker(save_path)
        
        # Create directories for saving models if they don't exist
        os.makedirs('ml_models', exist_ok=True)
        os.makedirs('cnn_models', exist_ok=True)
        
        # Print device information
        print("\n=== Device Information ===")
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("\n⚠️ WARNING: Running on CPU. This will be significantly slower!")
            print("💡 Tip: For faster processing, consider:")
            print("   1. Using a CUDA-capable GPU")
            print("   2. Installing PyTorch with CUDA support")
            print("   3. Ensuring your GPU drivers are up to date")

    @staticmethod
    def list_available_models():
        """List all available CNN and ML models."""
        # List CNN packages
        list_packages()
        
        # List ML Models
        print("\nML Classifiers:")
        for model in get_classifiers().keys():
            print(f"  • {model}")

    @staticmethod
    def get_transform():
        """Get the default transform for images."""
        return get_default_transform()
        
    def save_model(self, model, model_name, model_type, metrics=None):
        """Save trained model to disk."""
        if model_type == 'ml':
            save_path = os.path.join('ml_models', f"{model_name}.pkl")
            joblib.dump(model, save_path)
        elif model_type == 'cnn':
            save_path = os.path.join('cnn_models', f"{model_name}.pth")
            torch.save(model.state_dict(), save_path)
            
        if self.verbose:
            print(f"✅ Saved {model_type} model to {save_path}")
            if metrics:
                print(f"   Model accuracy: {metrics['Accuracy']:.4f}")
        
        return save_path

    def fit(self, train_loader, val_loader, cnn_models='biggest', ml_models=None):
        """Extract features using CNNs and evaluate multiple ML models."""
        # Get CNN models (either from package or specific list)
        try:
            cnn_models = get_cnn_models(cnn_models)
        except Exception as e:
            print(f"❌ Error with CNN models selection: {str(e)}")
            return None
        
        if ml_models is None:
            ml_models = list(self.classifiers.keys())
        
        print("\n=== Starting Feature Extraction and Training ===")
        print(f"Number of CNN models to try: {len(cnn_models)}")
        print(f"Number of ML models to try: {len(ml_models)}")
        print(f"Total combinations: {len(cnn_models) * len(ml_models)}")
        print(f"Models will be saved to: {self.models_dir}")
        
        start_time = time.time()
        best_ml_model = None
        best_accuracy = 0
        best_ml_name = None
        best_cnn_name = None

        for cnn_name in tqdm(cnn_models, desc="CNN Models"):
            if self.verbose:
                print(f"\n🔄 Extracting features using {cnn_name}...")
            
            try:
                # Extract features
                extractor = BaseCNNExtractor(cnn_name)
                
                # Save CNN feature extractor
                self.save_model(extractor.model, cnn_name, 'cnn')
                
                # Process training data
                train_features, train_labels = [], []
                for images, labels in tqdm(train_loader, desc="Training data", leave=False):
                    features = extractor.extract_features(images)
                    train_features.append(features.cpu().numpy())
                    train_labels.extend(labels.numpy())
                    
                train_features = np.concatenate(train_features)
                train_labels = np.array(train_labels)
                
                # Process validation data
                val_features, val_labels = [], []
                for images, labels in tqdm(val_loader, desc="Validation data", leave=False):
                    features = extractor.extract_features(images)
                    val_features.append(features.cpu().numpy())
                    val_labels.extend(labels.numpy())
                    
                val_features = np.concatenate(val_features)
                val_labels = np.array(val_labels)
                
                if self.verbose:
                    print(f"Feature shapes: Train {train_features.shape}, Val {val_features.shape}")
                
                # Try each ML model
                for ml_name in tqdm(ml_models, desc="ML Models", leave=False):
                    if self.verbose:
                        print(f"\n🔄 Training {ml_name}...")
                        
                    try:
                        # Train classifier
                        clf = self.classifiers[ml_name]
                        clf.fit(train_features, train_labels)
                        
                        # Calculate metrics
                        metrics = self.metrics.calculate_metrics(
                            clf, val_features, val_labels, 
                            cnn_name, ml_name, start_time
                        )
                        
                        # Save the trained ML model
                        model_name = f"{cnn_name}_{ml_name}"
                        self.save_model(clf, model_name, 'ml', metrics)
                        
                        # Track best model
                        if metrics and metrics['Accuracy'] > best_accuracy:
                            best_accuracy = metrics['Accuracy']
                            best_ml_model = clf
                            best_ml_name = ml_name
                            best_cnn_name = cnn_name
                        
                        # Print metrics
                        self.metrics.print_metrics(metrics, self.verbose)
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"❌ Error with {ml_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error with {cnn_name}: {str(e)}")
                continue
        
        # Print final results
        self.metrics.print_final_results()
        
        # Save best model separately
        if best_ml_model:
            best_model_path = self.save_model(
                best_ml_model, 
                f"best_{best_cnn_name}_{best_ml_name}", 
                'ml',
                {'Accuracy': best_accuracy}
            )
            print(f"\n✅ Best model saved to: {best_model_path}")
            
        return self.metrics.results

__version__ = "0.2.0"
__all__ = ['CNNFeatureExtractor', 'get_default_transform', 'load_custom_dataset'] 