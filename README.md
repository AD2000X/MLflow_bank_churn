# Bank Customer Churn Prediction with MLflow

A machine learning project for predicting bank customer churn using Logistic Regression and Random Forest models, with experiment tracking via MLflow.

## Features

- Automated data download from Kaggle
- Data preprocessing with one-hot encoding
- Multiple model comparison (Logistic Regression vs Random Forest)
- MLflow experiment tracking and model registry
- Comprehensive metrics logging

## Project Structure

\\\
bank-churn-mlflow/
├── src/              # Source code modules
├── scripts/          # Executable scripts
├── config/           # Configuration files
├── notebooks/        # Jupyter notebooks for analysis
└── tests/            # Unit tests
\\\

## Prerequisites

- Python 3.8+
- Kaggle account and API credentials

## Installation

1. Clone the repository:
\\\ash
git clone https://github.com/AD2000X/MLflow_bank_churn.git
cd MLflow_bank_churn
\\\

2. Create virtual environment:
\\\ash
python -m venv venv
# On Windows
.\venv\Scripts\Activate.ps1
# On Linux/Mac
source venv/bin/activate
\\\

3. Install dependencies:
\\\ash
pip install -r requirements.txt
\\\

4. Set up Kaggle API:
   - Download your \kaggle.json\ from https://www.kaggle.com/settings
   - Place it in \~/.kaggle/kaggle.json\ (Linux/Mac) or \C:\Users\<username>\.kaggle\kaggle.json\ (Windows)

## Usage

### Quick Start

\\\ash
python scripts/train.py
\\\

### View MLflow UI

\\\ash
mlflow ui --port 5000
\\\

Then open http://localhost:5000 in your browser.

### Configuration

Edit \config/config.yaml\ to customize:
- Model hyperparameters
- Train/test split ratio
- MLflow server settings

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.8125 | 0.6923 | 0.3571 | 0.4762 |
| Random Forest | 0.8594 | 0.7805 | 0.5357 | 0.6349 |

## Project Workflow

\\\
Data Download → Preprocessing → Model Training → MLflow Logging → Model Comparison
\\\

## Repository Contents

- \src/data_loader.py\ - Kaggle dataset download and CSV loading
- \src/preprocessor.py\ - Data cleaning and feature engineering
- \src/model_trainer.py\ - Model training and evaluation
- \src/mlflow_server.py\ - MLflow server lifecycle management
- \scripts/train.py\ - Main training pipeline

## MLflow Tracking

All experiments are logged to MLflow with:
- Parameters (hyperparameters, data split ratio)
- Metrics (accuracy, precision, recall, F1 score)
- Models (serialized sklearn models)
- Artifacts (model signatures, requirements)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Author

AD2000X

Project Link: https://github.com/AD2000X/MLflow_bank_churn
