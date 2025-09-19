"""
Data preprocessing module for customer churn prediction.
This module handles data loading, cleaning, and initial preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_clean_data(file_path):
    """
    Load and clean the customer churn dataset.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: Cleaned dataframe
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Drop customerID column as it's not needed for modeling
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # Replace blank TotalCharges with 0 and convert to float
    df['TotalCharges'] = df['TotalCharges'].replace({' ': '0'})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Convert target variable to binary (0: No, 1: Yes)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    
    return df


def get_feature_columns():
    """
    Return the feature column names for numerical and categorical features.
    
    Returns:
    tuple: (numerical_features, categorical_features)
    """
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    return numerical_features, categorical_features


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    
    Parameters:
    X (pd.DataFrame): Features
    y (pd.Series): Target variable
    test_size (float): Proportion of dataset to include in test split
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)