import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

class MetricsTracker:
    def __init__(self, save_path='results.csv'):
        self.save_path = save_path
        self.results = []
        
    def calculate_metrics(self, clf, val_features, val_labels, cnn_name, ml_name, start_time):
        """Calculate and store all metrics for a model."""
        try:
            val_preds = clf.predict(val_features)
            
            # Basic metrics
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            formatted_time = f"{int(minutes)} minutes {int(seconds)} sec"  # Format time
            
            metrics = {
                'CNN Model': cnn_name,
                'ML Model': ml_name,
                'Accuracy': clf.score(val_features, val_labels),
                'Balanced Accuracy': balanced_accuracy_score(val_labels, val_preds),
                'F1 Score': f1_score(val_labels, val_preds, average='weighted'),
                'Total Time': formatted_time  # Use formatted time
            }
            
            # ROC AUC (only if classifier supports predict_proba)
            try:
                val_probs = clf.predict_proba(val_features)
                metrics['ROC AUC'] = roc_auc_score(val_labels, val_probs, multi_class='ovr')
            except:
                metrics['ROC AUC'] = None
                
            self.results.append(metrics)
            self._save_results()
            return metrics
        except Exception as e:
            print(f"❌ Error calculating metrics for {cnn_name} + {ml_name}: {str(e)}")
            return None
    
    def _save_results(self):
        """Save current results to CSV file."""
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(self.save_path, index=False)
        except Exception as e:
            print(f"❌ Error saving results to {self.save_path}: {str(e)}")
        
    def get_best_model(self):
        """Get the best performing model based on accuracy."""
        if not self.results:
            return None
            
        df = pd.DataFrame(self.results)
        return df.loc[df['Accuracy'].idxmax()]
    
    def print_metrics(self, metrics, verbose=True):
        """Print metrics in a formatted way."""
        if not verbose or metrics is None:
            return
            
        print(f"\n✅ Model metrics:")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Balanced Accuracy: {metrics['Balanced Accuracy']:.4f}")
        print(f"   F1 Score: {metrics['F1 Score']:.4f}")
        if metrics['ROC AUC'] is not None:
            print(f"   ROC AUC: {metrics['ROC AUC']:.4f}")
        print(f"   Time: {metrics['Total Time']}")
        
    def print_final_results(self):
        """Print final results summary."""
        best_model = self.get_best_model()
        if best_model is None:
            print("\n❌ No results available.")
            return
            
        print("\n=== Training Complete ===")
        print(f"Best combination: {best_model['CNN Model']} + {best_model['ML Model']}")
        print(f"Best metrics:")
        print(f"  • Accuracy: {best_model['Accuracy']:.4f}")
        print(f"  • Balanced Accuracy: {best_model['Balanced Accuracy']:.4f}")
        print(f"  • F1 Score: {best_model['F1 Score']:.4f}")
        if best_model['ROC AUC'] is not None:
            print(f"  • ROC AUC: {best_model['ROC AUC']:.4f}")
        print(f"  • Total Time: {best_model['Total Time']} seconds") 