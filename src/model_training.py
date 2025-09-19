"""
Model training module for customer churn prediction.
This module handles model selection, hyperparameter tuning, and training with cross-validation.
"""

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def create_model_pipeline(preprocessor, model_type='random_forest', **model_params):
    """
    Create a full pipeline with preprocessing and classifier.
    
    Parameters:
    preprocessor (ColumnTransformer): Preprocessing pipeline
    model_type (str): Type of model to use ('random_forest' or 'xgboost')
    **model_params: Additional parameters for the model
    
    Returns:
    Pipeline: Full pipeline with preprocessing and classifier
    
    Documentation Reference:
    - Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    """
    from sklearn.pipeline import Pipeline
    
    # Set default parameters
    if model_type == 'random_forest':
        default_params = {'class_weight': 'balanced', 'random_state': 42}
        default_params.update(model_params)
        classifier = RandomForestClassifier(**default_params)
    elif model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier
            default_params = {'random_state': 42}
            default_params.update(model_params)
            classifier = XGBClassifier(**default_params)
        except ImportError:
            raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
    else:
        raise ValueError("Unsupported model type. Use 'random_forest' or 'xgboost'")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline


def get_hyperparameter_grid(model_type='random_forest'):
    """
    Get hyperparameter grid for GridSearchCV.
    
    Parameters:
    model_type (str): Type of model ('random_forest' or 'xgboost')
    
    Returns:
    dict: Hyperparameter grid
    """
    if model_type == 'random_forest':
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 20, None]
        }
    elif model_type == 'xgboost':
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [3, 6, 10]
        }
    else:
        raise ValueError("Unsupported model type. Use 'random_forest' or 'xgboost'")
    
    return param_grid


def train_model_with_cv(X_train, y_train, pipeline, param_grid, cv_folds=5, scoring='f1'):
    """
    Train model with cross-validation and hyperparameter tuning.
    
    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target
    pipeline (Pipeline): Model pipeline with preprocessing
    param_grid (dict): Hyperparameter grid for GridSearchCV
    cv_folds (int): Number of cross-validation folds
    scoring (str): Scoring metric for GridSearchCV
    
    Returns:
    GridSearchCV: Fitted GridSearchCV object
    
    Documentation Reference:
    - GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    """
    # Create stratified k-fold cross-validator
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Create GridSearchCV object
    search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1,  # Use all available processors
        verbose=1
    )
    
    # Fit the grid search
    search.fit(X_train, y_train)
    
    return search


def evaluate_model(model, X_test, y_test, class_names=['No Churn', 'Churn']):
    """
    Evaluate the trained model on test data.
    
    Parameters:
    model: Trained model
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test target
    class_names (list): Names of the classes for visualization
    
    Returns:
    dict: Evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create results dictionary
    results = {
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba
    }
    
    return results


def log_experiment_with_mlflow(search, X_test, y_test, experiment_name="customer_churn"):
    """
    Log experiment results with MLflow.
    
    Parameters:
    search (GridSearchCV): Fitted GridSearchCV object
    X_test (pd.DataFrame): Test features
    y_test (pd.Series): Test target
    experiment_name (str): Name of the MLflow experiment
    
    Documentation Reference:
    - mlflow.start_run: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
    - mlflow.log_params: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_params
    - mlflow.log_metric: https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric
    - mlflow.sklearn.log_model: https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model
    """
    try:
        # Set experiment name
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(search.best_params_)
            
            # Log best cross-validation score
            mlflow.log_metric("best_cv_f1_score", search.best_score_)
            
            # Evaluate on test set
            test_results = evaluate_model(search.best_estimator_, X_test, y_test)
            
            # Log test metrics
            mlflow.log_metric("test_f1_score", test_results['f1_score'])
            
            # Log classification report metrics
            for class_name in ['0', '1']:
                if class_name in test_results['classification_report']:
                    class_report = test_results['classification_report'][class_name]
                    mlflow.log_metric(f"test_precision_{class_name}", class_report['precision'])
                    mlflow.log_metric(f"test_recall_{class_name}", class_report['recall'])
                    mlflow.log_metric(f"test_f1_score_{class_name}", class_report['f1-score'])
            
            # Create input example for signature inference
            input_example = X_test.head(1)
            
            # Log model with signature and input example
            mlflow.sklearn.log_model(
                search.best_estimator_, 
                "model",
                input_example=input_example,
                conda_env={
                    "channels": ["defaults"],
                    "dependencies": [
                        "python=3.8",
                        "scikit-learn",
                        "pandas",
                        "numpy"
                    ],
                    "name": "customer_churn_env"
                }
            )
            
            # Log confusion matrix as artifact
            plt.figure(figsize=(8, 6))
            sns.heatmap(test_results['confusion_matrix'], annot=True, fmt='d', 
                        cmap='Blues', xticklabels=['No Churn', 'Churn'], 
                        yticklabels=['No Churn', 'Churn'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
            mlflow.log_artifact('confusion_matrix.png')
            plt.close()
            
            print(f"Experiment logged successfully with run ID: {run.info.run_id}")
            
    except Exception as e:
        print(f"Warning: Could not log experiment with MLflow: {e}")
        print("Continuing without MLflow logging...")


def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Parameters:
    model: Trained model pipeline
    filepath (str): Path to save the model
    """
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    filepath (str): Path to the saved model
    
    Returns:
    Model pipeline
    """
    return joblib.load(filepath)