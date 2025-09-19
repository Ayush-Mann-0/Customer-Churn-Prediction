"""
Training script for customer churn prediction model.
This script trains the model and saves it for API usage.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.insert(0, 'src')

from data_preprocessing import load_and_clean_data, get_feature_columns, split_data
from feature_engineering import create_preprocessor
from model_training import create_model_pipeline, get_hyperparameter_grid, train_model_with_cv, evaluate_model, log_experiment_with_mlflow, save_model


def main():
    """Main training function."""
    print("Customer Churn Prediction - Model Training")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('mlruns', exist_ok=True)
    
    # Load and clean data
    print("Loading and cleaning data...")
    try:
        df = load_and_clean_data('dataset.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: dataset.csv not found in the current directory.")
        print("Please ensure the dataset file is in the project root directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Separate features and target
    print("Separating features and target...")
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Churn distribution:\n{y.value_counts()}")
    
    # Get feature columns
    print("Getting feature columns...")
    numerical_features, categorical_features = get_feature_columns()
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Split data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create preprocessor
    print("Creating preprocessing pipeline...")
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    print("Preprocessing pipeline created successfully.")
    
    # Create model pipeline
    print("Creating model pipeline...")
    pipeline = create_model_pipeline(preprocessor, model_type='random_forest')
    print("Model pipeline created successfully.")
    
    # Get hyperparameter grid
    print("Getting hyperparameter grid...")
    param_grid = get_hyperparameter_grid('random_forest')
    print("Hyperparameter grid:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    
    # Train model with cross-validation
    print("Training model with cross-validation...")
    print("This may take a few minutes...")
    try:
        search = train_model_with_cv(
            X_train, y_train, pipeline, param_grid, 
            cv_folds=5, scoring='f1'
        )
        print("Model training completed successfully.")
        print(f"Best cross-validation F1 score: {search.best_score_:.4f}")
        print("Best parameters:")
        for key, value in search.best_params_.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    try:
        results = evaluate_model(search.best_estimator_, X_test, y_test)
        print(f"Test F1 Score: {results['f1_score']:.4f}")
        print("\nClassification Report:")
        # Convert classification report to DataFrame for better display
        report_df = pd.DataFrame(results['classification_report']).transpose()
        print(report_df)
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return
    
    # Log experiment with MLflow
    print("Logging experiment with MLflow...")
    try:
        log_experiment_with_mlflow(search, X_test, y_test, "customer_churn")
        print("Experiment logged successfully.")
    except Exception as e:
        print(f"Warning: Could not log experiment with MLflow: {e}")
        print("Continuing without MLflow logging...")
    
    # Save model
    print("Saving model...")
    try:
        save_model(search.best_estimator_, 'models/model.joblib')
        print("Model saved successfully to models/model.joblib")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # Create a simple success indicator
    with open('TRAINING_SUCCESS', 'w') as f:
        f.write(f"Training completed successfully!\n")
        f.write(f"Test F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"Model saved to: models/model.joblib\n")
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Next steps:")
    print("1. Start the API server: uvicorn src.api:app --host 0.0.0.0 --port 8000")
    print("2. View API documentation: http://localhost:8000/docs")
    print("3. View MLflow experiments: mlflow ui")
    print("=" * 50)


if __name__ == "__main__":
    main()