# Project Architecture and Design Decisions

This document explains the architectural decisions and design patterns used in the Customer Churn Prediction project.

## Architecture Overview

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Data Input    │───▶│  Preprocessing   │───▶│ Feature Eng.     │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                    │
                                                    ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Model Training │◀───│ Hyperparameter   │◀───│    Features      │
└─────────────────┘    │ Tuning           │    └──────────────────┘
         │             └──────────────────┘             │
         ▼                                            ▼
┌─────────────────┐                       ┌──────────────────────┐
│    Model        │◀──────────────────────│   Training Data      │
│   Evaluation    │                       └──────────────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────┐
│   Model Save    │───▶│    API Server    │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │    End Users     │
                      └──────────────────┘
```

## Design Decisions

### 1. Modular Structure

The project is organized into separate modules for different concerns:

- **`data_preprocessing.py`**: Handles data loading, cleaning, and initial transformations
- **`feature_engineering.py`**: Creates preprocessing pipelines for numerical and categorical features
- **`model_training.py`**: Implements model training, hyperparameter tuning, and evaluation
- **`api.py`**: Serves the trained model through a REST API

This separation provides:
- Reusability of components
- Easier testing and debugging
- Clear responsibility boundaries
- Better maintainability

### 2. Pipeline-based Preprocessing

We use scikit-learn's `Pipeline` and `ColumnTransformer` for preprocessing:

```python
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])
```

Benefits:
- Ensures consistent preprocessing between training and inference
- Prevents data leakage
- Simplifies model deployment
- Enables easy experimentation with different preprocessing steps

### 3. Model Selection

We chose Random Forest as the primary model for several reasons:

- Handles mixed data types well (numerical and categorical)
- Provides feature importance scores
- Robust to outliers
- Good performance on tabular data
- Interpretable results
- Built-in cross-validation support

### 4. Cross-Validation and Hyperparameter Tuning

We use `GridSearchCV` with stratified k-fold cross-validation:

```python
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
```

Benefits:
- More robust model evaluation
- Prevents overfitting to a specific train/test split
- Automatic hyperparameter optimization
- Better generalization performance

### 5. API Design with FastAPI

FastAPI was chosen for the API implementation because:

- High performance (ASGI server)
- Automatic API documentation (Swagger/OpenAPI)
- Type validation with Pydantic
- Asynchronous support
- Easy to deploy and scale

### 6. Model Serialization with Joblib

We use joblib for model serialization instead of pickle because:

- More efficient for numpy arrays and scikit-learn models
- Better compression
- Faster serialization/deserialization
- Designed specifically for scientific computing

### 7. Experiment Tracking with MLflow

MLflow is integrated for:

- Tracking hyperparameters and metrics
- Versioning trained models
- Comparing different experiments
- Reproducibility

### 8. Docker Containerization

Docker is used for:

- Consistent deployment environments
- Easy scaling
- Simplified dependency management
- Platform independence

## Key Technical Choices

### 1. Feature Engineering Strategy

We use different preprocessing for numerical and categorical features:

- **Numerical features**: Median imputation + Standard scaling
- **Categorical features**: Most frequent imputation + One-hot encoding

### 2. Data Validation

Pydantic models ensure data validation at the API level:

```python
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    # ... other fields
```

### 3. Error Handling

Comprehensive error handling with appropriate HTTP status codes:

- 422 for validation errors
- 500 for internal server errors
- Custom error messages for debugging

### 4. Testing Strategy

We implement unit tests for each module:

- Test data preprocessing functions
- Test feature engineering pipelines
- Test model training components
- Test API endpoints

### 5. Performance Considerations

- Use of `n_jobs=-1` in GridSearchCV for parallel processing
- Efficient data structures (pandas DataFrames)
- Batch prediction support in the API
- Memory-efficient model serialization

## Scalability Considerations

### Horizontal Scaling

The API can be scaled horizontally by:
- Running multiple instances behind a load balancer
- Using gunicorn with multiple workers
- Container orchestration with Kubernetes or Docker Swarm

### Vertical Scaling

For larger datasets:
- Implement data streaming
- Use distributed computing frameworks (Dask, Spark)
- Optimize database queries if using external data sources

## Security Considerations

- Input validation at the API level
- No sensitive data stored in logs
- Container isolation with Docker
- Proper error handling to prevent information leakage

## Future Improvements

1. **Model Monitoring**: Implement model performance monitoring in production
2. **A/B Testing**: Support for testing multiple model versions
3. **Advanced Models**: Integration with XGBoost, LightGBM, or deep learning models
4. **Real-time Feature Store**: Integration with feature stores for real-time features
5. **CI/CD Pipeline**: Automated testing and deployment
6. **Rate Limiting**: Implement rate limiting for API endpoints
7. **Authentication**: Add authentication for API access