import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, List, Tuple, Any
import joblib
import json
import sys
from pathlib import Path
from backend.models.similarity_model import SimilarityModel


# Add project root to sys.path for module imports
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Now import your custom model
from models.similarity_model import SimilarityModel


class ModelTrainer:
    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {
            'tfidf': SimilarityModel(model_type='tfidf'),
            'bert': SimilarityModel(model_type='bert'),
            'spacy': SimilarityModel(model_type='spacy')
        }
        
        # Initialize classifiers
        self.classifiers = {
            'logistic_regression': LogisticRegression(max_iter=1000),
            'svm': SVC(probability=True)
        }
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        # Combine skills and experience into a single text
        df['combined_text'] = df.apply(
            lambda x: f"{' '.join(x['skills'])} {x['raw_text']}", axis=1
        )
        
        X = df['combined_text'].values
        y = df['job_title'].values
        
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train all models and return their performance metrics."""
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and evaluate each model
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name} model...")
            
            # Get embeddings
            X_train_emb = np.array([model.get_embedding(text) for text in X_train])
            X_test_emb = np.array([model.get_embedding(text) for text in X_test])
            
            # Train and evaluate each classifier
            for clf_name, clf in self.classifiers.items():
                print(f"Training {clf_name}...")
                
                # Train classifier
                clf.fit(X_train_emb, y_train)
                
                # Make predictions
                y_pred = clf.predict(X_test_emb)
                
                # Calculate metrics
                report = classification_report(y_test, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y_test, y_pred)
                
                # Save results
                results[f"{model_name}_{clf_name}"] = {
                    'classification_report': report,
                    'confusion_matrix': conf_matrix.tolist()
                }
                
                # Save model
                model_path = self.model_dir / f"{model_name}_{clf_name}.joblib"
                joblib.dump(clf, model_path)
        
        # Save results
        with open(self.model_dir / "training_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate all trained models."""
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Get embeddings
            X_emb = np.array([model.get_embedding(text) for text in X])
            
            for clf_name, clf in self.classifiers.items():
                print(f"Evaluating {clf_name}...")
                
                # Load model if not already loaded
                model_path = self.model_dir / f"{model_name}_{clf_name}.joblib"
                if not hasattr(clf, 'predict'):
                    clf = joblib.load(model_path)
                
                # Make predictions
                y_pred = clf.predict(X_emb)
                
                # Calculate metrics
                report = classification_report(y, y_pred, output_dict=True)
                conf_matrix = confusion_matrix(y, y_pred)
                
                # Save results
                results[f"{model_name}_{clf_name}"] = {
                    'classification_report': report,
                    'confusion_matrix': conf_matrix.tolist()
                }
        
        # Save results
        with open(self.model_dir / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def compare_models(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Compare performance of different models."""
        comparison_data = []
        
        for model_name, metrics in results.items():
            report = metrics['classification_report']
            
            # Extract relevant metrics
            comparison_data.append({
                'Model': model_name,
                'Accuracy': report['accuracy'],
                'Macro F1': report['macro avg']['f1-score'],
                'Weighted F1': report['weighted avg']['f1-score']
            })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_df.to_csv(self.model_dir / "model_comparison.csv", index=False)
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Load processed data
    df = pd.read_csv(Path("data/processed/tagged_resumes.csv"))
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Train models
    results = trainer.train_models(X, y)
    
    # Evaluate models
    eval_results = trainer.evaluate_models(X, y)
    
    # Compare models
    comparison = trainer.compare_models(results)
    print("\nModel Comparison:")
    print(comparison)
