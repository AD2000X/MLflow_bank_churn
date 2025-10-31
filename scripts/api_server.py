"""
Simple Flask API Server for Bank Churn Prediction
Usage: python scripts/api_server.py

This script provides a REST API for making churn predictions using
the latest trained model from MLflow Model Registry.
"""

from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import sys
import os
from datetime import datetime

# Add src to path for potential imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)

# Configuration
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MODEL_NAME = os.getenv('MODEL_NAME', 'BankChurnRandomForest')

# Global variables for model
model = None
model_version = None
model_info = {}


def load_model():
    """Load the latest model from MLflow Registry"""
    global model, model_version, model_info
    
    print(f"Connecting to MLflow at {MLFLOW_URI}...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    
    print(f"Loading model: {MODEL_NAME}")
    
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    if not versions:
        raise ValueError(f"No model found with name '{MODEL_NAME}'")
    
    # Get the latest version
    latest = max(versions, key=lambda v: int(v.version))
    model_version = latest.version
    
    # Load the model
    model_uri = f"models:/{MODEL_NAME}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Store model info
    model_info = {
        'name': MODEL_NAME,
        'version': model_version,
        'stage': latest.current_stage,
        'run_id': latest.run_id,
        'loaded_at': datetime.now().isoformat()
    }
    
    print(f"??Model loaded successfully!")
    print(f"  Name: {MODEL_NAME}")
    print(f"  Version: {model_version}")
    print(f"  Stage: {latest.current_stage}")
    print(f"  Run ID: {latest.run_id}\n")


# Load model when starting the app
try:
    load_model()
except Exception as e:
    print(f"??Error loading model: {e}")
    print("\nMake sure:")
    print("1. MLflow server is running")
    print("2. Models are trained and registered")
    print(f"3. Model '{MODEL_NAME}' exists in MLflow\n")
    sys.exit(1)


@app.route('/', methods=['GET'])
def home():
    """
    Health check endpoint
    Returns: JSON with server status and model info
    """
    return jsonify({
        'status': 'running',
        'message': 'Bank Churn Prediction API',
        'model': model_info,
        'endpoints': {
            'health': '/ (GET)',
            'predict': '/predict (POST)',
            'model_info': '/model-info (GET)',
            'reload': '/reload-model (POST)'
        }
    })


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get detailed model information
    Returns: JSON with complete model metadata
    """
    return jsonify(model_info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make churn prediction for a customer
    
    Expected JSON format:
    {
        "CreditScore": 650,
        "Age": 35,
        "Tenure": 5,
        "Balance": 50000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 75000
    }
    
    Returns: JSON with prediction result
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'example': {
                    'CreditScore': 650,
                    'Age': 35,
                    'Tenure': 5,
                    'Balance': 50000,
                    'NumOfProducts': 2,
                    'HasCrCard': 1,
                    'IsActiveMember': 1,
                    'EstimatedSalary': 75000
                }
            }), 400
        
        # Convert to DataFrame with float type
        input_df = pd.DataFrame([data]).astype(float).astype(float)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_int = int(prediction)
        
        # Prepare response
        result = {
            'prediction': prediction_int,
            'prediction_label': 'WILL CHURN' if prediction_int == 1 else 'WILL STAY',
            'confidence': 'Model provides binary classification without probability',
            'model_version': model_version,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field: {str(e)}',
            'required_fields': [
                'CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
            ]
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/reload-model', methods=['POST'])
def reload_model():
    """
    Reload the latest model without restarting the server
    Useful when a new model version is trained
    
    Returns: JSON with reload status
    """
    try:
        print("\nReloading model...")
        load_model()
        
        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'model': model_info
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple customers at once
    
    Expected JSON format:
    {
        "customers": [
            {"CreditScore": 650, "Age": 35, ...},
            {"CreditScore": 720, "Age": 45, ...},
            ...
        ]
    }
    
    Returns: JSON with predictions for all customers
    """
    try:
        data = request.get_json()
        
        if not data or 'customers' not in data:
            return jsonify({
                'error': 'Invalid format. Expected: {"customers": [...]}'
            }), 400
        
        customers = data['customers']
        
        if not isinstance(customers, list) or len(customers) == 0:
            return jsonify({
                'error': 'customers must be a non-empty list'
            }), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame(customers)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Prepare results
        results = []
        for i, pred in enumerate(predictions):
            pred_int = int(pred)
            results.append({
                'customer_index': i,
                'prediction': pred_int,
                'prediction_label': 'WILL CHURN' if pred_int == 1 else 'WILL STAY'
            })
        
        # Calculate statistics
        churn_count = sum(1 for p in predictions if p == 1)
        churn_rate = churn_count / len(predictions) * 100
        
        return jsonify({
            'total_customers': len(predictions),
            'churn_count': churn_count,
            'churn_rate': f"{churn_rate:.1f}%",
            'predictions': results,
            'model_version': model_version,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction error: {str(e)}'
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': {
            'health': '/ (GET)',
            'predict': '/predict (POST)',
            'batch_predict': '/batch-predict (POST)',
            'model_info': '/model-info (GET)',
            'reload': '/reload-model (POST)'
        }
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500


def main():
    """Main function to run the Flask app"""
    print("=" * 70)
    print("BANK CHURN PREDICTION API SERVER")
    print("=" * 70)
    print(f"\nMLflow Tracking URI: {MLFLOW_URI}")
    print(f"Model: {MODEL_NAME} v{model_version}")
    print("\nStarting server on http://0.0.0.0:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              - Health check")
    print("  GET  /model-info    - Model information")
    print("  POST /predict       - Single prediction")
    print("  POST /batch-predict - Batch predictions")
    print("  POST /reload-model  - Reload latest model")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        use_reloader=False  # Disable reloader to prevent double model loading
    )


if __name__ == '__main__':
    main()
