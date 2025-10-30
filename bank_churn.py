# Bank Customer Churn Prediction with MLflow

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import kagglehub
import threading
import time
import psutil

# 1. Download Dataset

print("=" * 70)
print("[Step 1] Downloading Dataset")
print("=" * 70)

path = kagglehub.dataset_download("harshitstark/bank-churn-train")
print(f"[Path] {path}")

files = os.listdir(path)
print(f"[Files] {files}")


# 2. Load Data

print("\n" + "=" * 70)
print("[Step 2] Loading Data")
print("=" * 70)

csv_file = os.path.join(path, "train.csv")
df = pd.read_csv(csv_file)

print(f"[Info] Records: {len(df)}, Columns: {len(df.columns)}")


# 3. Data Preprocessing

print("\n" + "=" * 70)
print("[Step 3] Data Preprocessing")
print("=" * 70)

print(f"[Shape] Original: {df.shape}")

# Drop unnecessary columns
columns_to_drop = ['id', 'CustomerId', 'Surname']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print(f"[Info] Dropped columns: {columns_to_drop}")

# Check missing values
missing = df.isnull().sum().sum()
print(f"[Info] Missing values: {missing}")

# Check target variable distribution
print(f"[Info] Churn rate: {df['Exited'].mean():.2%}")

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)
print(f"[Info] After encoding: {df.shape}")

# Separate features and target
X = df.drop(columns=['Exited']).astype(float)
y = df['Exited']
print(f"[Info] Features: {X.shape[1]}, Samples: {X.shape[0]}")


# 4. Start MLflow Server

print("\n" + "=" * 70)
print("[Step 4] Starting MLflow Server")
print("=" * 70)

# Terminate existing MLflow processes
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = proc.info.get('cmdline')
        if cmdline and 'mlflow' in ' '.join(cmdline):
            proc.kill()
            print("[Info] Terminated existing MLflow process")
    except:
        pass

time.sleep(2)

# Start MLflow server in background
def start_mlflow_server():
    os.system("mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns")

mlflow_thread = threading.Thread(target=start_mlflow_server, daemon=True)
mlflow_thread.start()

print("[Info] Waiting for MLflow server to start...")
time.sleep(15)

print("[URL] http://localhost:5000")

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bank Churn Prediction")
print("[Info] MLflow ready")


# 5. Split Data

print("\n" + "=" * 70)
print("[Step 5] Splitting Data")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[Info] Train set: {X_train.shape}, Test set: {X_test.shape}")


# 6. Train Logistic Regression

print("\n" + "=" * 70)
print("[Step 6] Training Logistic Regression")
print("=" * 70)

with mlflow.start_run(run_name="logistic-regression-baseline") as run:
    # Train model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = log_reg.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Display results
    print("\n[Results] Logistic Regression:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Log to MLflow
    mlflow.log_params({
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "test_size": 0.2,
        "random_state": 42,
        "n_features": X_train.shape[1]
    })

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    signature = infer_signature(X_train, log_reg.predict(X_train))
    mlflow.sklearn.log_model(
        log_reg, "model",
        signature=signature,
        registered_model_name="BankChurnLogisticRegression"
    )

    print(f"[Info] Logged to MLflow (Run ID: {run.info.run_id})")


# 7. Train Random Forest

print("\n" + "=" * 70)
print("[Step 7] Training Random Forest")
print("=" * 70)

with mlflow.start_run(run_name="random-forest-model") as run:
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)

    # Calculate metrics
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)

    # Display results
    print("\n[Results] Random Forest:")
    print(f"  Accuracy:  {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
    print(f"  Precision: {precision_rf:.4f}")
    print(f"  Recall:    {recall_rf:.4f}")
    print(f"  F1 Score:  {f1_rf:.4f}")

    # Log to MLflow
    mlflow.log_params({
        "model_type": "RandomForest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    mlflow.log_metrics({
        "accuracy": accuracy_rf,
        "precision": precision_rf,
        "recall": recall_rf,
        "f1_score": f1_rf
    })

    signature = infer_signature(X_train, rf_model.predict(X_train))
    mlflow.sklearn.log_model(
        rf_model, "model",
        signature=signature,
        registered_model_name="BankChurnRandomForest"
    )

    print(f"[Info] Logged to MLflow (Run ID: {run.info.run_id})")


# 8. Summary

print("\n" + "=" * 70)
print("[Completed]")
print("=" * 70)
print(f"\n[Comparison] Model Results:")
print(f"  Logistic Regression - F1: {f1:.4f}")
print(f"  Random Forest       - F1: {f1_rf:.4f}")
print(f"\n[MLflow UI] http://localhost:5000")
print("\n[Info] MLflow server running in background")
print("       Press Ctrl+C to stop")

# Keep program running
try:
    print("\n[Info] Program running... (Press Ctrl+C to stop)")
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n\n[Info] Program terminated")
