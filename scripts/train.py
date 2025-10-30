"""
Main training script.
Orchestrates the complete ML pipeline.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader
from preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from mlflow_server import MLflowServer
import mlflow


def main():
    """Execute complete training pipeline."""
    
    # Initialize components
    loader = DataLoader()
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    server = MLflowServer(port=5000)
    
    # Start MLflow server
    server.start()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Load data
    df = loader.load_csv()
    
    # Preprocess
    X, y = preprocessor.preprocess(df)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train models
    trainer = ModelTrainer()
    metrics_lr = trainer.train_logistic_regression(X_train, X_test, y_train, y_test)
    metrics_rf = trainer.train_random_forest(X_train, X_test, y_train, y_test)
    
    # Summary
    print("\n" + "=" * 70)
    print("[Completed]")
    print("=" * 70)
    print(f"\n[Comparison] Model Results:")
    print(f"  Logistic Regression - F1: {metrics_lr['f1_score']:.4f}")
    print(f"  Random Forest       - F1: {metrics_rf['f1_score']:.4f}")
    print(f"\n[MLflow UI] http://localhost:5000")
    print("\n[Info] MLflow server running in background")
    print("       Press Ctrl+C to stop")
    
    # Keep running
    try:
        print("\n[Info] Program running... (Press Ctrl+C to stop)")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\n[Info] Program terminated")


if __name__ == "__main__":
    main()
