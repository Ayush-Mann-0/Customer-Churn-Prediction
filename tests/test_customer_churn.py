"""
Test module for customer churn prediction components.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_and_clean_data, get_feature_columns, split_data
from feature_engineering import create_preprocessor
from model_training import create_model_pipeline, get_hyperparameter_grid


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data_preprocessing module."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'customerID': ['1', '2', '3'],
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No'],
            'tenure': [12, 24, 36],
            'PhoneService': ['Yes', 'No', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No'],
            'InternetService': ['DSL', 'Fiber optic', 'DSL'],
            'OnlineSecurity': ['Yes', 'No', 'Yes'],
            'OnlineBackup': ['No', 'Yes', 'No'],
            'DeviceProtection': ['Yes', 'No', 'Yes'],
            'TechSupport': ['No', 'Yes', 'No'],
            'StreamingTV': ['Yes', 'No', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No'],
            'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Electronic check'],
            'MonthlyCharges': [29.85, 56.95, 53.85],
            'TotalCharges': ['29.85', '1889.5', '108.15'],
            'Churn': ['No', 'Yes', 'No']
        })

    def test_load_and_clean_data(self):
        """Test load_and_clean_data function."""
        # Save sample data to CSV
        test_file = 'test_data.csv'
        self.sample_data.to_csv(test_file, index=False)
        
        # Load and clean data
        cleaned_data = load_and_clean_data(test_file)
        
        # Check that customerID is dropped
        self.assertNotIn('customerID', cleaned_data.columns)
        
        # Check that TotalCharges is converted to float
        self.assertEqual(cleaned_data['TotalCharges'].dtype, np.float64)
        
        # Check that Churn is converted to binary
        self.assertIn(0, cleaned_data['Churn'].unique())
        self.assertIn(1, cleaned_data['Churn'].unique())
        
        # Clean up
        os.remove(test_file)

    def test_get_feature_columns(self):
        """Test get_feature_columns function."""
        numerical_features, categorical_features = get_feature_columns()
        
        # Check that we have the expected number of features
        self.assertEqual(len(numerical_features), 3)
        self.assertEqual(len(categorical_features), 16)
        
        # Check for specific features
        self.assertIn('tenure', numerical_features)
        self.assertIn('gender', categorical_features)

    def test_split_data(self):
        """Test split_data function."""
        # Create sample data for testing
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # Check sizes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature_engineering module."""

    def setUp(self):
        """Set up test data."""
        self.numerical_features = ['feature1', 'feature2']
        self.categorical_features = ['cat1', 'cat2']
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0],
            'feature2': [5.0, np.nan, 7.0, 8.0],
            'cat1': ['A', 'B', 'A', 'C'],
            'cat2': ['X', 'Y', 'X', 'Z']
        })

    def test_create_preprocessor(self):
        """Test create_preprocessor function."""
        preprocessor = create_preprocessor(self.numerical_features, self.categorical_features)
        
        # Check that preprocessor is a ColumnTransformer
        from sklearn.compose import ColumnTransformer
        self.assertIsInstance(preprocessor, ColumnTransformer)
        
        # Check that it has the expected transformers by checking the transformers list
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        self.assertIn('num', transformer_names)
        self.assertIn('cat', transformer_names)


class TestModelTraining(unittest.TestCase):
    """Test cases for model_training module."""

    def test_create_model_pipeline(self):
        """Test create_model_pipeline function."""
        # Create a simple preprocessor for testing
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Simple preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), [0])
            ]
        )
        
        # Test with random forest
        pipeline_rf = create_model_pipeline(preprocessor, model_type='random_forest')
        self.assertIsInstance(pipeline_rf, Pipeline)
        
        # Test with xgboost
        try:
            pipeline_xgb = create_model_pipeline(preprocessor, model_type='xgboost')
            self.assertIsInstance(pipeline_xgb, Pipeline)
        except ImportError:
            # XGBoost not installed, skip this test
            pass

    def test_get_hyperparameter_grid(self):
        """Test get_hyperparameter_grid function."""
        # Test with random forest
        rf_grid = get_hyperparameter_grid('random_forest')
        self.assertIn('classifier__n_estimators', rf_grid)
        self.assertIn('classifier__max_depth', rf_grid)
        
        # Test with xgboost
        xgb_grid = get_hyperparameter_grid('xgboost')
        self.assertIn('classifier__n_estimators', xgb_grid)
        self.assertIn('classifier__max_depth', xgb_grid)


if __name__ == '__main__':
    unittest.main()