"""
Feature engineering module for customer churn prediction.
This module builds the preprocessing pipeline using sklearn's ColumnTransformer and Pipeline.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


def create_preprocessor(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    Parameters:
    numerical_features (list): List of numerical feature column names
    categorical_features (list): List of categorical feature column names
    
    Returns:
    ColumnTransformer: Preprocessing pipeline
    
    Documentation Reference:
    - ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
    - Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
    - OneHotEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    - SimpleImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    """
    
    # Preprocessing for numerical features
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor


def get_feature_names_after_preprocessing(preprocessor, numerical_features, categorical_features, X_sample):
    """
    Get feature names after preprocessing.
    
    Parameters:
    preprocessor (ColumnTransformer): Fitted preprocessor
    numerical_features (list): List of numerical feature names
    categorical_features (list): List of categorical feature names
    X_sample (pd.DataFrame): Sample data to fit the preprocessor
    
    Returns:
    list: List of feature names after preprocessing
    """
    # Fit the preprocessor on sample data
    preprocessor.fit(X_sample)
    
    # Get feature names for numerical features (they remain the same)
    numerical_names = numerical_features
    
    # Get feature names for categorical features
    categorical_names = preprocessor.named_transformers_['cat']\
        .named_steps['encoder'].get_feature_names_out(categorical_features)
    
    # Combine all feature names
    all_feature_names = list(numerical_names) + list(categorical_names)
    
    return all_feature_names