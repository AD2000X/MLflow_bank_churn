"""
Model training module.
Handles model training, evaluation, and MLflow logging.
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    """Train and evaluate ML models with MLflow tracking."""
    
    def __init__(self, experiment_name="Bank Churn Prediction"):
        """
        Initialize model trainer.
        
        Args:
            experiment_name: MLflow experiment name
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """
        Train Logistic Regression model.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            dict: Model metrics
        """
        print("\n" + "=" * 70)
        print("[Step 6] Training Logistic Regression")
        print("=" * 70)
        
        with mlflow.start_run(run_name="logistic-regression-baseline") as run:
            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Display results
            self._print_results("Logistic Regression", metrics)
            
            # Log to MLflow
            mlflow.log_params({
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "test_size": 0.2,
                "random_state": 42,
                "n_features": X_train.shape[1]
            })
            
            mlflow.log_metrics(metrics)
            
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                registered_model_name="BankChurnLogisticRegression"
            )
            
            print(f"[Info] Logged to MLflow (Run ID: {run.info.run_id})")
            
            return metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """
        Train Random Forest model.
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data
            
        Returns:
            dict: Model metrics
        """
        print("\n" + "=" * 70)
        print("[Step 7] Training Random Forest")
        print("=" * 70)
        
        with mlflow.start_run(run_name="random-forest-model") as run:
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Display results
            self._print_results("Random Forest", metrics)
            
            # Log to MLflow
            mlflow.log_params({
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            })
            
            mlflow.log_metrics(metrics)
            
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model, "model",
                signature=signature,
                registered_model_name="BankChurnRandomForest"
            )
            
            print(f"[Info] Logged to MLflow (Run ID: {run.info.run_id})")
            
            return metrics
    
    @staticmethod
    def _calculate_metrics(y_true, y_pred):
        """Calculate classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }
    
    @staticmethod
    def _print_results(model_name, metrics):
        """Print model evaluation results."""
        print(f"\n[Results] {model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
