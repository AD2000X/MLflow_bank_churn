"""
Model Deployment Scenarios for Bank Churn Prediction
Demonstrates various ways to deploy and use the trained MLflow models
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# CORE FUNCTIONS: Model Loading
# ============================================================================

def load_latest_model(model_name, tracking_uri="http://127.0.0.1:5000", stage=None):
    """
    Load the latest version of a registered model from MLflow
    
    Args:
        model_name: Name of the registered model
        tracking_uri: MLflow tracking server URI
        stage: Model stage filter (None, "Staging", "Production", "Archived")
               If None, gets the latest version regardless of stage
    
    Returns:
        Loaded model and version info
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    try:
        if stage:
            # Get latest version in specific stage
            versions = client.get_latest_versions(model_name, stages=[stage])
        else:
            # Get all versions and find the latest
            all_versions = client.search_model_versions(f"name='{model_name}'")
            if not all_versions:
                raise ValueError(f"No versions found for model '{model_name}'")
            versions = [max(all_versions, key=lambda v: int(v.version))]
        
        if not versions:
            raise ValueError(f"No model found for '{model_name}' with stage '{stage}'")
        
        latest_version = versions[0]
        version_number = latest_version.version
        
        print(f"Loading model: {model_name}")
        print(f"Version: {version_number}")
        print(f"Stage: {latest_version.current_stage}")
        print(f"Run ID: {latest_version.run_id}")
        
        # Load the model
        model_uri = f"models:/{model_name}/{version_number}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        print(f"??Model loaded successfully!\n")
        
        return model, {
            'version': version_number,
            'stage': latest_version.current_stage,
            'run_id': latest_version.run_id,
            'model_uri': model_uri
        }
    
    except Exception as e:
        print(f"??Error loading model: {e}")
        raise


def load_specific_version(model_name, version, tracking_uri="http://127.0.0.1:5000"):
    """
    Load a specific version of a registered model
    
    Args:
        model_name: Name of the registered model
        version: Version number to load
        tracking_uri: MLflow tracking server URI
    
    Returns:
        Loaded model
    """
    mlflow.set_tracking_uri(tracking_uri)
    
    model_uri = f"models:/{model_name}/{version}"
    print(f"Loading {model_name} version {version}...")
    
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"??Model version {version} loaded successfully!\n")
    
    return model


# ============================================================================
# SCENARIO 1: Simple Deployment - Load and Predict
# ============================================================================

def scenario_1_simple_deployment():
    """
    Basic deployment: Load latest model and make predictions
    """
    print("=" * 70)
    print("SCENARIO 1: SIMPLE DEPLOYMENT")
    print("=" * 70 + "\n")
    
    # Load the latest Random Forest model
    model, model_info = load_latest_model("BankChurnRandomForest")
    
    # Create sample customer data for prediction
    sample_customer = pd.DataFrame([{
        'CreditScore': 650.0,
        'Age': 35.0,
        'Tenure': 5.0,
        'Balance': 50000.0,
        'NumOfProducts': 2.0,
        'HasCrCard': 1.0,
        'IsActiveMember': 1.0,
        'EstimatedSalary': 75000.0,
        'Geography_Germany': 0.0,
        'Geography_Spain': 0.0,
        'Gender_Male': 1.0
    }])
    
    print("Sample customer data:")
    print(sample_customer.T)
    print()
    
    # Make prediction
    prediction = model.predict(sample_customer)[0]
    
    print(f"Prediction: {'WILL CHURN' if prediction == 1 else 'WILL STAY'}")
    print(f"Prediction value: {prediction}")
    print(f"Model version used: {model_info['version']}\n")


# ============================================================================
# SCENARIO 2: Flask API Server with Latest Model
# ============================================================================

def scenario_2_flask_api():
    """
    Create a Flask API that automatically loads the latest model on startup
    """
    from flask import Flask, request, jsonify
    
    print("=" * 70)
    print("SCENARIO 2: FLASK API WITH AUTO-RELOAD")
    print("=" * 70 + "\n")
    
    # Configuration
    MODEL_NAME = "BankChurnRandomForest"
    TRACKING_URI = "http://127.0.0.1:5000"
    
    def create_app():
        """Initialize Flask app with latest model"""
        app = Flask(__name__)
        
        # Load model on startup
        print("Initializing Flask API...")
        model, model_info = load_latest_model(MODEL_NAME, TRACKING_URI)
        
        # Store model info in app config
        app.config['MODEL'] = model
        app.config['MODEL_INFO'] = model_info
        app.config['MODEL_NAME'] = MODEL_NAME
        
        @app.route('/', methods=['GET'])
        def home():
            """Health check endpoint"""
            return jsonify({
                'status': 'running',
                'model_name': app.config['MODEL_NAME'],
                'model_version': app.config['MODEL_INFO']['version'],
                'model_stage': app.config['MODEL_INFO']['stage'],
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/model-info', methods=['GET'])
        def model_info_endpoint():
            """Get detailed model information"""
            return jsonify(app.config['MODEL_INFO'])
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """Make predictions"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Convert to DataFrame
                input_df = pd.DataFrame([data])
                
                # Make prediction
                prediction = app.config['MODEL'].predict(input_df)[0]
                
                return jsonify({
                    'prediction': int(prediction),
                    'prediction_label': 'WILL CHURN' if prediction == 1 else 'WILL STAY',
                    'model_version': app.config['MODEL_INFO']['version'],
                    'timestamp': datetime.now().isoformat()
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/reload-model', methods=['POST'])
        def reload_model():
            """Reload the latest model without restarting the server"""
            try:
                print("\nReloading model...")
                model, model_info = load_latest_model(MODEL_NAME, TRACKING_URI)
                
                app.config['MODEL'] = model
                app.config['MODEL_INFO'] = model_info
                
                return jsonify({
                    'status': 'success',
                    'message': 'Model reloaded successfully',
                    'new_version': model_info['version']
                })
            
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
        
        return app
    
    # Create and return the app
    app = create_app()
    
    print("\nFlask API created successfully!")
    print("To run the server, execute:")
    print("  app.run(host='0.0.0.0', port=5000)")
    print("\nAPI Endpoints:")
    print("  GET  /              - Health check")
    print("  GET  /model-info    - Model information")
    print("  POST /predict       - Make predictions")
    print("  POST /reload-model  - Reload latest model")
    print()
    
    return app


# ============================================================================
# SCENARIO 3: Batch Prediction
# ============================================================================

def scenario_3_batch_prediction(input_csv_path=None):
    """
    Run batch predictions on a dataset using the latest model
    
    Args:
        input_csv_path: Path to CSV file with customer data
    """
    print("=" * 70)
    print("SCENARIO 3: BATCH PREDICTION")
    print("=" * 70 + "\n")
    
    # Load the latest model
    model, model_info = load_latest_model("BankChurnRandomForest")
    
    # Generate or load batch data
    if input_csv_path:
        print(f"Loading data from {input_csv_path}...")
        batch_data = pd.read_csv(input_csv_path)
    else:
        print("Generating sample batch data...")
        # Generate sample data
        np.random.seed(42)
        n_samples = 100
        
        batch_data = pd.DataFrame({
            'CreditScore': np.random.randint(300, 850, n_samples).astype(float),
            'Age': np.random.randint(18, 70, n_samples).astype(float),
            'Tenure': np.random.randint(0, 11, n_samples).astype(float),
            'Balance': np.random.uniform(0, 250000, n_samples),
            'NumOfProducts': np.random.randint(1, 5, n_samples).astype(float),
            'HasCrCard': np.random.randint(0, 2, n_samples).astype(float),
            'IsActiveMember': np.random.randint(0, 2, n_samples).astype(float),
            'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
            'Geography_Germany': np.random.randint(0, 2, n_samples).astype(float),
            'Geography_Spain': np.random.randint(0, 2, n_samples).astype(float),
            'Gender_Male': np.random.randint(0, 2, n_samples).astype(float)
        })
    
    print(f"Batch data shape: {batch_data.shape}")
    print(f"First few rows:\n{batch_data.head()}\n")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(batch_data)
    
    # Add predictions to dataframe
    batch_data['Prediction'] = predictions
    batch_data['Prediction_Label'] = batch_data['Prediction'].map({
        0: 'WILL STAY',
        1: 'WILL CHURN'
    })
    
    # Calculate statistics
    churn_rate = (predictions == 1).sum() / len(predictions) * 100
    
    print(f"??Predictions completed!")
    print(f"Total customers: {len(predictions)}")
    print(f"Predicted to churn: {(predictions == 1).sum()} ({churn_rate:.1f}%)")
    print(f"Predicted to stay: {(predictions == 0).sum()} ({100-churn_rate:.1f}%)")
    print(f"Model version used: {model_info['version']}\n")
    
    # Save results
    output_path = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    batch_data.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}\n")
    
    return batch_data


# ============================================================================
# SCENARIO 4: A/B Testing - Compare Model Versions
# ============================================================================

def scenario_4_ab_testing():
    """
    Compare predictions between different model versions or different models
    """
    print("=" * 70)
    print("SCENARIO 4: A/B TESTING - MODEL COMPARISON")
    print("=" * 70 + "\n")
    
    # Load different models/versions for comparison
    print("Loading models for comparison...\n")
    
    # Model A: Logistic Regression (latest version)
    model_lr, info_lr = load_latest_model("BankChurnLogisticRegression")
    
    # Model B: Random Forest (latest version)
    model_rf, info_rf = load_latest_model("BankChurnRandomForest")
    
    # Generate test data
    print("Generating test data...")
    np.random.seed(42)
    n_samples = 50
    
    test_data = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n_samples).astype(float),
        'Age': np.random.randint(18, 70, n_samples).astype(float),
        'Tenure': np.random.randint(0, 11, n_samples).astype(float),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.randint(1, 5, n_samples).astype(float),
        'HasCrCard': np.random.randint(0, 2, n_samples).astype(float),
        'IsActiveMember': np.random.randint(0, 2, n_samples).astype(float),
        'EstimatedSalary': np.random.uniform(10000, 200000, n_samples),
        'Geography_Germany': np.random.randint(0, 2, n_samples).astype(float),
        'Geography_Spain': np.random.randint(0, 2, n_samples).astype(float),
        'Gender_Male': np.random.randint(0, 2, n_samples).astype(float)
    })
    
    print(f"Test data shape: {test_data.shape}\n")
    
    # Make predictions with both models
    print("Making predictions with both models...")
    predictions_lr = model_lr.predict(test_data)
    predictions_rf = model_rf.predict(test_data)
    
    # Compare results
    comparison_df = pd.DataFrame({
        'Customer_ID': range(1, n_samples + 1),
        'LR_Prediction': predictions_lr,
        'RF_Prediction': predictions_rf,
        'Agreement': predictions_lr == predictions_rf
    })
    
    # Calculate statistics
    agreement_rate = (predictions_lr == predictions_rf).sum() / n_samples * 100
    lr_churn_rate = (predictions_lr == 1).sum() / n_samples * 100
    rf_churn_rate = (predictions_rf == 1).sum() / n_samples * 100
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nModel A: Logistic Regression (v{info_lr['version']})")
    print(f"  Predicted churn rate: {lr_churn_rate:.1f}%")
    print(f"\nModel B: Random Forest (v{info_rf['version']})")
    print(f"  Predicted churn rate: {rf_churn_rate:.1f}%")
    print(f"\nAgreement between models: {agreement_rate:.1f}%")
    print(f"Disagreement cases: {(~comparison_df['Agreement']).sum()}")
    
    # Show disagreement cases
    disagreements = comparison_df[~comparison_df['Agreement']]
    if len(disagreements) > 0:
        print(f"\nFirst 5 disagreement cases:")
        print(disagreements.head())
    
    # Save comparison results
    output_path = f"ab_test_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nComparison results saved to: {output_path}\n")
    
    return comparison_df


# ============================================================================
# SCENARIO 5: Model Version Comparison
# ============================================================================

def scenario_5_version_comparison(model_name="BankChurnRandomForest"):
    """
    Compare different versions of the same model
    """
    print("=" * 70)
    print("SCENARIO 5: MODEL VERSION COMPARISON")
    print("=" * 70 + "\n")
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    
    # Get all versions of the model
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    if len(all_versions) < 2:
        print(f"Only {len(all_versions)} version(s) found for {model_name}")
        print("Need at least 2 versions for comparison.\n")
        return
    
    print(f"Found {len(all_versions)} versions of {model_name}")
    print("Comparing version 1 vs latest version...\n")
    
    # Load version 1 and latest version
    model_v1 = load_specific_version(model_name, version=1)
    model_latest = load_specific_version(model_name, version=len(all_versions))
    
    # Generate test data
    np.random.seed(42)
    test_data = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, 30).astype(float),
        'Age': np.random.randint(18, 70, 30).astype(float),
        'Tenure': np.random.randint(0, 11, 30).astype(float),
        'Balance': np.random.uniform(0, 250000, 30),
        'NumOfProducts': np.random.randint(1, 5, 30).astype(float),
        'HasCrCard': np.random.randint(0, 2, 30).astype(float),
        'IsActiveMember': np.random.randint(0, 2, 30).astype(float),
        'EstimatedSalary': np.random.uniform(10000, 200000, 30),
        'Geography_Germany': np.random.randint(0, 2, 30).astype(float),
        'Geography_Spain': np.random.randint(0, 2, 30).astype(float),
        'Gender_Male': np.random.randint(0, 2, 30).astype(float)
    })
    
    # Make predictions
    pred_v1 = model_v1.predict(test_data)
    pred_latest = model_latest.predict(test_data)
    
    # Compare
    agreement = (pred_v1 == pred_latest).sum() / len(pred_v1) * 100
    
    print("=" * 70)
    print(f"Version 1 churn rate: {(pred_v1 == 1).sum() / len(pred_v1) * 100:.1f}%")
    print(f"Latest version churn rate: {(pred_latest == 1).sum() / len(pred_latest) * 100:.1f}%")
    print(f"Agreement: {agreement:.1f}%\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to demonstrate all deployment scenarios
    """
    print("\n" + "=" * 70)
    print("BANK CHURN PREDICTION - MODEL DEPLOYMENT SCENARIOS")
    print("=" * 70 + "\n")
    
    print("This script demonstrates 5 different deployment scenarios:")
    print("1. Simple deployment with prediction")
    print("2. Flask API with auto-reload")
    print("3. Batch prediction")
    print("4. A/B testing between models")
    print("5. Version comparison")
    print("\n" + "=" * 70 + "\n")
    
    # Run all scenarios
    try:
        # Scenario 1
        scenario_1_simple_deployment()
        input("Press Enter to continue to Scenario 2...")
        
        # Scenario 2
        app = scenario_2_flask_api()
        print("Note: Flask app created but not started.")
        print("To start it, uncomment the app.run() line at the end of this script.\n")
        input("Press Enter to continue to Scenario 3...")
        
        # Scenario 3
        scenario_3_batch_prediction()
        input("Press Enter to continue to Scenario 4...")
        
        # Scenario 4
        scenario_4_ab_testing()
        input("Press Enter to continue to Scenario 5...")
        
        # Scenario 5
        scenario_5_version_comparison()
        
        print("\n" + "=" * 70)
        print("ALL SCENARIOS COMPLETED!")
        print("=" * 70 + "\n")
    
    except Exception as e:
        print(f"\n??Error: {e}")
        print("\nMake sure:")
        print("1. MLflow server is running at http://127.0.0.1:5000")
        print("2. Models are trained and registered")
        print("3. Run 'python bank_churn_mlflow.py' first to train models\n")


if __name__ == "__main__":
    # Run all scenarios
    main()
    
    # Uncomment to run individual scenarios:
    # scenario_1_simple_deployment()
    # scenario_3_batch_prediction()
    # scenario_4_ab_testing()
    # scenario_5_version_comparison()
    
    # Uncomment to start Flask API:
    # app = scenario_2_flask_api()
    # app.run(host='0.0.0.0', port=5000, debug=False)

